#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

bool HwBenchEnabled() noexcept {
  static const bool enabled = [] {
    const char* v = std::getenv("GUFO_STRIX_HW_BENCH");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return enabled;
}

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
};

/// The trunk and the optional MTP block as one traversal's output.
struct Uploaded {
  DeviceTensor token_embd;
  DeviceTensor output;
  DeviceMixer hc_head;
  std::vector<DeviceLayer> layers;
  DeviceLayer mtp;
};

/// Walks the model's tensors in upload order. With `weights` null the walk
/// only measures: it sums the bytes every allocation would request and does
/// no device work, so the weight placement can be decided from the tensors.
struct Uploader {
  hip::WeightUpload& stager;
  const hip::WeightAllocator* weights;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};
  std::uint64_t requested{0};

  [[nodiscard]] bool Measuring() const noexcept { return weights == nullptr; }

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  [[nodiscard]] hipError_t Allocate(void** pointer, std::size_t size) {
    requested += size;
    if (Measuring()) {
      return hipSuccess;
    }
    return weights->Allocate(pointer, size);
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("device allocation failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    const auto combined = Copy(t);
    if (!ok || (!Measuring() && !stager.Finish(error))) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (Allocate(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      if (Measuring()) {
        continue;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    if (Measuring()) {
      return;
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0)) {
        Fail("stacked upload needs F32 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (!Measuring()) {
      allocations.push_back(ptr);
      bytes += size + kTailMargin;
      std::size_t offset = 0;
      for (const TensorRef* t : parts) {
        if (!stager.Copy(shard_base + t->shard, t->file_offset, t->SizeBytes(),
                         static_cast<std::uint8_t*>(ptr) + offset, error)) {
          Fail("upload failed for " + std::string(t->name));
          return d;
        }
        offset += t->SizeBytes();
      }
    }
    if (type == core::GgmlType::kQ8_0) {
      if (Measuring()) {
        return d;
      }
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (Allocate(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({ptr, half, count});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  DeviceMixer Mixer(const HcMixer& m) {
    return {Copy(m.norm), Copy(m.down), Copy(m.up), Copy(m.inject)};
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || t->type != core::GgmlType::kQ8_0 ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = Copy(l.ssm_qkv);
      d.ssm_gate = Copy(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.linear) {
      d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta});
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = Copy(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = Copy(l.attn_q);
      d.attn_k = Copy(l.attn_k);
      d.attn_v = Copy(l.attn_v);
    }
    d.attn_out = Copy(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = Copy(l.indexer_q);
    d.indexer_k = Copy(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = Copy(l.ple_key);
    d.ple_value = Copy(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = Copy(l.ple_conv1d);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }

  Uploaded Model(const ModelWeights& w, const MtpWeights* mtp,
                 std::uint32_t trunk_shards) {
    Uploaded u;
    u.token_embd = Copy(w.token_embd);
    u.output =
        w.output.data == w.token_embd.data ? u.token_embd : Copy(w.output);
    u.hc_head = Mixer(w.hc_head);
    u.layers.reserve(w.layers.size());
    for (const auto& l : w.layers) {
      u.layers.push_back(Layer(l));
      if (!ok) {
        return u;
      }
    }
    if (mtp != nullptr) {
      shard_base = trunk_shards;
      u.mtp = Layer(mtp->block);
    }
    return u;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

bool DeviceModel::SetDraftVocab(std::uint32_t prefix_rows,
                                std::vector<std::int32_t> specials,
                                std::string* error_msg) {
  const auto fail = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return false;
  };
  if (output_.empty() || output_.rows == 0 || output_bytes_ % output_.rows != 0)
    return fail("draft vocabulary needs a row-addressable output head");
  if (prefix_rows == 0 || prefix_rows >= output_.rows)
    return fail("draft vocabulary prefix must leave rows out");
  for (const auto id : specials) {
    if (id < static_cast<std::int32_t>(prefix_rows) ||
        static_cast<std::uint32_t>(id) >= output_.rows)
      return fail("draft special tokens must lie above the prefix");
  }
  // The prefix is the head's own leading rows: no copy unless the local Q4_K
  // draft head experiment is switched on.
  draft_output_ = output_;
  draft_output_.rows = prefix_rows;
  if (const char* q4k_draft = std::getenv("GUFO_STRIX_Q4K_DRAFT");
      q4k_draft != nullptr && q4k_draft[0] != '\0' && q4k_draft[0] != '0' &&
      output_.type == core::GgmlType::kQ8_0 && output_.cols % 256 == 0) {
    const std::size_t size = std::size_t{prefix_rows} * (output_.cols / 256) * 144;
    void* q4k = nullptr;
    if (hipMalloc(&q4k, size + kTailMargin) != hipSuccess)
      return fail("device allocation failed for the Q4_K draft head");
    allocations_.push_back(q4k);
    bytes_ += size + kTailMargin;
    QuantizeQ8_0ToQ4K(output_.data, q4k, prefix_rows, output_.cols, nullptr);
    if (hipMemsetAsync(static_cast<std::uint8_t*>(q4k) + size, 0, kTailMargin,
                       nullptr) != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess)
      return fail("Q4_K draft head quantization failed");
    draft_output_.q4k = q4k;
    std::fprintf(stderr, "strix-q4k-draft: %u rows, %zu MiB\n", prefix_rows,
                 size >> 20);
  }
  if (HwBenchEnabled() && draft_output_.q4k != nullptr &&
      draft_output_.cols >= 256 && draft_output_.cols % 256 == 0) {
    const int K = static_cast<int>(draft_output_.cols);
    const std::size_t q8_bytes = qfn_mmq_q8_1_bytes(3, K);
    float* d_x = nullptr;
    void* d_q8 = nullptr;
    float* d_out = nullptr;
    if (hip::AllocateDevice(&d_x, std::size_t{3} * K * sizeof(float)) ==
            hipSuccess &&
        hip::AllocateDevice(&d_q8, q8_bytes) == hipSuccess &&
        hip::AllocateDevice(&d_out, std::size_t{3} * prefix_rows *
                                        sizeof(float)) == hipSuccess) {
      std::vector<float> h_x(std::size_t{3} * K, 0.01f);
      for (std::size_t i = 0; i < h_x.size(); ++i)
        h_x[i] = static_cast<float>((i % 31) - 15) * 0.015625f;
      if (hipMemcpyAsync(d_x, h_x.data(), h_x.size() * sizeof(float),
                         hipMemcpyHostToDevice, nullptr) == hipSuccess &&
          qfn_mmq_quantize_q8_1(d_x, d_q8, 3, K, nullptr) == 0 &&
          hipDeviceSynchronize() == hipSuccess) {
        hipEvent_t e0 = nullptr, e1 = nullptr;
        if (hipEventCreate(&e0) == hipSuccess &&
            hipEventCreate(&e1) == hipSuccess) {
          std::vector<std::uint32_t> row_counts;
          for (std::uint32_t r : {32768U, 49152U, 65536U, prefix_rows}) {
            if (r <= prefix_rows &&
                std::find(row_counts.begin(), row_counts.end(), r) ==
                    row_counts.end())
              row_counts.push_back(r);
          }
          for (std::uint32_t R : row_counts) {
            const double bytes =
                static_cast<double>(R) * (K / 256) * 144.0;
            for (int M : {1, 2, 3}) {
              for (int w = 0; w < 3; ++w)
                (void)qfn_mmq_q4_K_dense_vec_preq(
                    draft_output_.q4k, nullptr, d_q8, d_out,
                    static_cast<int>(R), M, K, nullptr);
              float t[10]{};
              bool ok = true;
              for (int i = 0; i < 10; ++i) {
                ok = ok && hipEventRecord(e0, nullptr) == hipSuccess &&
                     qfn_mmq_q4_K_dense_vec_preq(
                         draft_output_.q4k, nullptr, d_q8, d_out,
                         static_cast<int>(R), M, K, nullptr) == 0 &&
                     hipEventRecord(e1, nullptr) == hipSuccess &&
                     hipEventSynchronize(e1) == hipSuccess &&
                     hipEventElapsedTime(&t[i], e0, e1) == hipSuccess;
              }
              if (ok) {
                std::sort(t, t + 10);
                const double med = 0.5 * (t[4] + t[5]);
                const double gbs = med > 0.0 ? bytes / (med * 1e-3) / 1e9 : 0.0;
                std::fprintf(
                    stderr,
                    "strix-hw-bench: q4k_draft rows=%u M=%d ms=%.3f gbs=%.1f\n",
                    R, M, med, gbs);
              }
            }
          }
        }
        if (e0) (void)hipEventDestroy(e0);
        if (e1) (void)hipEventDestroy(e1);
      }
    } else {
      (void)hipGetLastError();
    }
    if (d_x) (void)hipFree(d_x);
    if (d_q8) (void)hipFree(d_q8);
    if (d_out) (void)hipFree(d_out);
  }
  draft_specials_ = std::move(specials);
  return true;
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg) {
  // The CPU reference also reads Q6_K, but the production embedding, dense
  // and routed kernels do not. Reject it before allocating device weights.
  const auto supported = [&](const TensorRef& t) {
    if (t.type != core::GgmlType::kQ6_K)
      return true;
    if (error_msg != nullptr)
      *error_msg = "unsupported HIP tensor format Q6_K: " + std::string(t.name);
    return false;
  };
  const auto layer_supported = [&](const LayerWeights& l) {
    return supported(l.ffn_gate_exps) && supported(l.ffn_up_exps) &&
           supported(l.ffn_down_exps);
  };
  if (!supported(w.token_embd) || !supported(w.output) ||
      !std::all_of(w.layers.begin(), w.layers.end(), layer_supported) ||
      (mtp != nullptr && !layer_supported(mtp->block))) {
    return nullptr;
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  // Measure first: the placement is decided once, from the bytes this walk
  // will request of the device heap.
  std::uint64_t measured = 0;
  {
    std::vector<Conversion> conversions;
    std::vector<void*> allocations;
    std::size_t bytes = 0;
    std::size_t max_half_cols = 1;
    std::size_t max_q8_cols = 32;
    Uploader measure{*stager, nullptr,       conversions, allocations,
                     bytes,   max_half_cols, max_q8_cols, error_msg};
    (void)measure.Model(w, mtp, shard_count);
    if (!measure.ok) {
      return nullptr;
    }
    measured = measure.requested;
  }
  const hip::WeightAllocator weights(measured);
  std::vector<Conversion> conversions;
  Uploader up{*stager,   &weights,          conversions,     m->allocations_,
              m->bytes_, m->max_half_cols_, m->max_q8_cols_, error_msg};
  Uploaded uploaded = up.Model(w, mtp, shard_count);
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  assert(up.requested == measured);
  m->token_embd_ = uploaded.token_embd;
  m->output_ = uploaded.output;
  m->output_bytes_ = w.output.SizeBytes();
  m->hc_head_ = uploaded.hc_head;
  m->layers_ = std::move(uploaded.layers);
  m->mtp_ = uploaded.mtp;
  m->has_mtp_ = mtp != nullptr;
  for (const auto& c : conversions) {
    NarrowActivations(static_cast<const float*>(c.source), c.destination, false,
                      c.count, nullptr);
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.count * sizeof(float) + kTailMargin;
  }
  // Local experiment: GUFO_STRIX_Q4K_DENSE lists projections (ssm_in,
  // ssm_out, attn_qkv) that get a Q4_K copy for small decode batches. The
  // Q8_0 originals stay for prefill and wide batches.
  if (const char* list = std::getenv("GUFO_STRIX_Q4K_DENSE");
      list != nullptr && list[0] != '\0') {
    std::size_t copies = 0;
    std::size_t copy_bytes = 0;
    bool failed = false;
    const auto shadow = [&](DeviceTensor& t, const char* name) {
      if (failed || std::strstr(list, name) == nullptr || t.empty() ||
          t.type != core::GgmlType::kQ8_0 || t.cols % 256 != 0)
        return;
      const std::size_t size = std::size_t{t.rows} * (t.cols / 256) * 144;
      void* q4k = nullptr;
      if (hipMalloc(&q4k, size + kTailMargin) != hipSuccess) {
        failed = true;
        return;
      }
      m->allocations_.push_back(q4k);
      m->bytes_ += size + kTailMargin;
      QuantizeQ8_0ToQ4K(t.data, q4k, t.rows, t.cols, nullptr);
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(q4k) + size, 0,
                           kTailMargin, nullptr);
      t.q4k = q4k;
      ++copies;
      copy_bytes += size;
    };
    for (auto& l : m->layers_) {
      shadow(l.ssm_in, "ssm_in");
      shadow(l.ssm_out, "ssm_out");
      shadow(l.attn_qkv, "attn_qkv");
      shadow(l.attn_out, "attn_out");
      shadow(l.shexp_up, "shexp");
      shadow(l.shexp_gate, "shexp");
      shadow(l.hc_attn.down, "hc_down");
      shadow(l.hc_ffn.down, "hc_down");
    }
    void* ids = nullptr;
    constexpr std::size_t kIdBytes = 64 * sizeof(std::int32_t);
    if (failed || hipMalloc(&ids, kIdBytes) != hipSuccess ||
        hipMemsetAsync(ids, 0, kIdBytes, nullptr) != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess) {
      if (error_msg != nullptr)
        *error_msg = "Q4_K dense copy failed";
      return nullptr;
    }
    m->allocations_.push_back(ids);
    m->q4k_ids_ = static_cast<std::int32_t*>(ids);
    std::fprintf(stderr, "strix-q4k: %zu dense copies, %zu MiB (%s)\n", copies,
                 copy_bytes >> 20, list);
  }
  if (const char* q4k_mtp = std::getenv("GUFO_STRIX_Q4K_MTP");
      m->has_mtp_ && q4k_mtp != nullptr && q4k_mtp[0] != '\0' &&
      q4k_mtp[0] != '0') {
    std::size_t dense_copies = 0;
    std::size_t expert_tensors = 0;
    std::size_t total_bytes = 0;
    bool failed = false;
    const auto shadow_mtp = [&](DeviceTensor& t) {
      if (failed || t.empty() || t.type != core::GgmlType::kQ8_0 ||
          t.cols % 256 != 0)
        return;
      const std::size_t size = std::size_t{t.rows} * (t.cols / 256) * 144;
      void* q4k = nullptr;
      if (hipMalloc(&q4k, size + kTailMargin) != hipSuccess) {
        failed = true;
        return;
      }
      m->allocations_.push_back(q4k);
      m->bytes_ += size + kTailMargin;
      QuantizeQ8_0ToQ4K(t.data, q4k, t.rows, t.cols, nullptr);
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(q4k) + size, 0,
                           kTailMargin, nullptr);
      t.q4k = q4k;
      ++dense_copies;
      total_bytes += size;
    };
    const auto convert_mtp_exps = [&](DeviceTensor& t) {
      if (failed || t.empty() || t.type != core::GgmlType::kQ8_0 ||
          t.cols % 256 != 0 || t.experts == 0)
        return;
      const std::size_t total_rows = std::size_t{t.experts} * t.rows;
      const std::size_t size = total_rows * (t.cols / 256) * 144;
      void* q4k = nullptr;
      if (hipMalloc(&q4k, size + kTailMargin) != hipSuccess) {
        failed = true;
        return;
      }
      m->allocations_.push_back(q4k);
      m->bytes_ += size + kTailMargin;
      QuantizeQ8_0ToQ4K(t.data, q4k, total_rows, t.cols, nullptr);
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(q4k) + size, 0,
                           kTailMargin, nullptr);
      t.data = q4k;
      t.type = core::GgmlType::kQ4_K;
      ++expert_tensors;
      total_bytes += size;
    };
    shadow_mtp(m->mtp_.attn_qkv);
    shadow_mtp(m->mtp_.attn_out);
    shadow_mtp(m->mtp_.nextn_fc_embedding);
    shadow_mtp(m->mtp_.nextn_fc_hidden);
    shadow_mtp(m->mtp_.hc_attn.down);
    shadow_mtp(m->mtp_.hc_ffn.down);
    shadow_mtp(m->mtp_.nextn_head.down);
    shadow_mtp(m->mtp_.shexp_up);
    shadow_mtp(m->mtp_.shexp_gate);
    convert_mtp_exps(m->mtp_.ffn_gate_exps);
    convert_mtp_exps(m->mtp_.ffn_up_exps);
    if (failed || hipDeviceSynchronize() != hipSuccess) {
      if (error_msg != nullptr)
        *error_msg = "Q4_K MTP conversion failed";
      return nullptr;
    }
    std::fprintf(stderr,
                 "strix-q4k-mtp: %zu dense copies, %zu expert tensors, %zu MiB\n",
                 dense_copies, expert_tensors, total_bytes >> 20);
  }
  if (const char* mtp_down = std::getenv("GUFO_STRIX_MTP_DOWN_IQ4");
      m->has_mtp_ && mtp_down != nullptr && mtp_down[0] != '\0' &&
      mtp_down[0] != '0') {
    auto& t = m->mtp_.ffn_down_exps;
    if (!t.empty() && t.cols % 32 == 0 && t.experts > 0 &&
        (t.type == core::GgmlType::kQ8_0 || t.type == core::GgmlType::kQ5_1)) {
      const std::size_t total_rows = std::size_t{t.experts} * t.rows;
      const std::size_t old_block_bytes =
          t.type == core::GgmlType::kQ8_0 ? 34 : 24;
      const std::size_t old_alloc_bytes =
          total_rows * (t.cols / 32) * old_block_bytes + kTailMargin;
      const std::size_t dst_size = total_rows * (t.cols / 32) * 18;
      void* dst = nullptr;
      if (weights.Allocate(&dst, dst_size + kTailMargin) != hipSuccess) {
        if (error_msg != nullptr)
          *error_msg = "MTP down IQ4_NL conversion allocation failed";
        return nullptr;
      }
      QuantizeExpsToIQ4NL(t.data, t.type, dst, total_rows, t.cols, nullptr);
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(dst) + dst_size, 0,
                           kTailMargin, nullptr);
      if (hipDeviceSynchronize() != hipSuccess) {
        (void)hipFree(dst);
        if (error_msg != nullptr)
          *error_msg = "MTP down IQ4_NL conversion failed";
        return nullptr;
      }
      void* old_ptr = t.data;
      std::erase(m->allocations_, old_ptr);
      (void)hipFree(old_ptr);
      m->allocations_.push_back(dst);
      m->bytes_ -= old_alloc_bytes;
      m->bytes_ += dst_size + kTailMargin;
      t.data = dst;
      t.type = core::GgmlType::kIQ4_NL;
      std::fprintf(stderr, "strix-mtp-down: IQ4_NL (saved %zu MiB)\n",
                   (old_alloc_bytes - (dst_size + kTailMargin)) >> 20);
    }
  }
  if (const char* moe_down = std::getenv("GUFO_STRIX_MOE_DOWN");
      moe_down != nullptr && moe_down[0] != '\0' && moe_down[0] != '0') {
    const bool to_mxfp4 = std::strncmp(moe_down, "mxfp4", 5) == 0;
    const bool to_iq4nl = std::strncmp(moe_down, "iq4_nl", 6) == 0;
    const bool only_q5 = std::strstr(moe_down, "_q5") != nullptr;
    if (!to_mxfp4 && !to_iq4nl) {
      if (error_msg != nullptr)
        *error_msg = "invalid GUFO_STRIX_MOE_DOWN: " + std::string(moe_down);
      return nullptr;
    }
    const core::GgmlType dst_type =
        to_mxfp4 ? core::GgmlType::kMXFP4 : core::GgmlType::kIQ4_NL;
    const std::size_t dst_block_bytes = to_mxfp4 ? 17 : 18;
    std::size_t converted_layers = 0;
    std::size_t freed_bytes = 0;
    std::size_t new_bytes = 0;
    for (auto& l : m->layers_) {
      auto& t = l.ffn_down_exps;
      if (t.empty() || t.cols % 32 != 0 || t.experts == 0)
        continue;
      if (only_q5 && t.type != core::GgmlType::kQ5_1)
        continue;
      if (!only_q5 && t.type != core::GgmlType::kQ5_1 &&
          t.type != core::GgmlType::kQ8_0)
        continue;
      const std::size_t total_rows = std::size_t{t.experts} * t.rows;
      const std::size_t old_block_bytes =
          t.type == core::GgmlType::kQ8_0 ? 34 : 24;
      const std::size_t old_alloc_bytes =
          total_rows * (t.cols / 32) * old_block_bytes + kTailMargin;
      const std::size_t dst_size =
          total_rows * (t.cols / 32) * dst_block_bytes;
      void* dst = nullptr;
      if (weights.Allocate(&dst, dst_size + kTailMargin) != hipSuccess) {
        if (error_msg != nullptr)
          *error_msg = "MoE down 4-bit conversion allocation failed";
        return nullptr;
      }
      if (to_mxfp4)
        QuantizeExpsToMXFP4(t.data, t.type, dst, total_rows, t.cols, nullptr);
      else
        QuantizeExpsToIQ4NL(t.data, t.type, dst, total_rows, t.cols, nullptr);
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(dst) + dst_size, 0,
                           kTailMargin, nullptr);
      if (hipDeviceSynchronize() != hipSuccess) {
        (void)hipFree(dst);
        if (error_msg != nullptr)
          *error_msg = "MoE down 4-bit conversion failed";
        return nullptr;
      }
      void* old_ptr = t.data;
      std::erase(m->allocations_, old_ptr);
      (void)hipFree(old_ptr);
      m->allocations_.push_back(dst);
      m->bytes_ -= old_alloc_bytes;
      m->bytes_ += dst_size + kTailMargin;
      t.data = dst;
      t.type = dst_type;
      ++converted_layers;
      freed_bytes += old_alloc_bytes;
      new_bytes += dst_size + kTailMargin;
    }
    std::fprintf(stderr, "strix-moe-down: %zu layers -> %s (saved %zu MiB)\n",
                 converted_layers, to_mxfp4 ? "MXFP4" : "IQ4_NL",
                 (freed_bytes - new_bytes) >> 20);
  }
  if (const char* selftest = std::getenv("GUFO_STRIX_Q4K_SELFTEST");
      selftest != nullptr && selftest[0] != '\0' && selftest[0] != '0') {
    const auto run_4bit_selftest = [&]() -> bool {
      struct Q5_1BlockHost {
        std::uint16_t d;
        std::uint16_t m;
        std::uint32_t qh;
        std::uint8_t qs[16];
      };
      struct Iq4NlBlockHost {
        std::uint16_t d;
        std::uint8_t qs[16];
      };
      struct Mxfp4BlockHost {
        std::uint8_t e;
        std::uint8_t qs[16];
      };
      static constexpr std::int8_t kMxfp4Table[16] = {
          0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
      };
      const auto e8m0_to_f32 = [](std::uint8_t e) -> float {
        const std::uint32_t bits =
            e == 0 ? 0x00400000U : (static_cast<std::uint32_t>(e) << 23U);
        float f = 0.0f;
        std::memcpy(&f, &bits, sizeof(float));
        return f;
      };
      constexpr int M = 64;
      constexpr int K = 640;
      constexpr int E = 4;
      constexpr int B = K / 32;
      constexpr int T_max = 40;
      const std::size_t q5_bytes = std::size_t{E} * M * B * sizeof(Q5_1BlockHost);
      const std::size_t iq4_bytes =
          std::size_t{E} * M * B * sizeof(Iq4NlBlockHost);
      const std::size_t mx_bytes =
          std::size_t{E} * M * B * sizeof(Mxfp4BlockHost);
      std::vector<Q5_1BlockHost> h_q5(E * M * B);
      constexpr std::uint16_t kDs[4] = {0x3000, 0x2C00, 0xB000, 0x3400};
      constexpr std::uint16_t kMs[4] = {0xB800, 0xB400, 0x3400, 0x0000};
      for (int idx = 0; idx < E * M * B; ++idx) {
        auto& blk = h_q5[idx];
        blk.d = kDs[idx & 3];
        blk.m = kMs[(idx >> 2) & 3];
        blk.qh = static_cast<std::uint32_t>(idx * 2654435761U);
        for (int j = 0; j < 16; ++j)
          blk.qs[j] = static_cast<std::uint8_t>((idx * 17 + j * 31) & 0xFF);
      }
      // Exact Q8_1-representable inputs (block scale = 1/128 = 0x2000).
      std::vector<float> h_x(static_cast<std::size_t>(T_max) * K);
      for (int t = 0; t < T_max; ++t) {
        for (int b = 0; b < B; ++b) {
          h_x[static_cast<std::size_t>(t) * K + b * 32 + 0] =
              127.0f * 0.0078125f;
          for (int i = 1; i < 32; ++i) {
            const int q = ((i * 7 + b * 13 + t * 19) % 121) - 60;
            h_x[static_cast<std::size_t>(t) * K + b * 32 + i] =
                static_cast<float>(q) * 0.0078125f;
          }
        }
      }
      std::vector<std::int32_t> h_ids(T_max);
      for (int t = 0; t < T_max; ++t)
        h_ids[t] = (t * 3 + (t / 4)) % E;

      void* d_q5 = nullptr;
      void* d_iq4 = nullptr;
      void* d_mx = nullptr;
      float* d_x = nullptr;
      std::int32_t* d_ids = nullptr;
      float* d_out = nullptr;
      const auto cleanup = [&]() {
        if (d_q5) (void)hipFree(d_q5);
        if (d_iq4) (void)hipFree(d_iq4);
        if (d_mx) (void)hipFree(d_mx);
        if (d_x) (void)hipFree(d_x);
        if (d_ids) (void)hipFree(d_ids);
        if (d_out) (void)hipFree(d_out);
      };
      if (hip::AllocateDevice(&d_q5, q5_bytes + kTailMargin) != hipSuccess ||
          hip::AllocateDevice(&d_iq4, iq4_bytes + kTailMargin) != hipSuccess ||
          hip::AllocateDevice(&d_mx, mx_bytes + kTailMargin) != hipSuccess ||
          hip::AllocateDevice(&d_x, h_x.size() * sizeof(float)) != hipSuccess ||
          hip::AllocateDevice(&d_ids, T_max * sizeof(std::int32_t)) !=
              hipSuccess ||
          hip::AllocateDevice(&d_out, std::size_t{T_max} * M * sizeof(float)) !=
              hipSuccess) {
        cleanup();
        return false;
      }
      bool ok =
          hipMemcpyAsync(d_q5, h_q5.data(), q5_bytes, hipMemcpyHostToDevice,
                         nullptr) == hipSuccess &&
          hipMemsetAsync(static_cast<std::uint8_t*>(d_iq4) + iq4_bytes, 0,
                         kTailMargin, nullptr) == hipSuccess &&
          hipMemsetAsync(static_cast<std::uint8_t*>(d_mx) + mx_bytes, 0,
                         kTailMargin, nullptr) == hipSuccess &&
          hipMemcpyAsync(d_x, h_x.data(), h_x.size() * sizeof(float),
                         hipMemcpyHostToDevice, nullptr) == hipSuccess &&
          hipMemcpyAsync(d_ids, h_ids.data(), T_max * sizeof(std::int32_t),
                         hipMemcpyHostToDevice, nullptr) == hipSuccess;
      std::vector<Iq4NlBlockHost> h_iq4(E * M * B);
      std::vector<Mxfp4BlockHost> h_mx(E * M * B);
      if (ok) {
        QuantizeExpsToIQ4NL(d_q5, core::GgmlType::kQ5_1, d_iq4, E * M, K,
                            nullptr);
        QuantizeExpsToMXFP4(d_q5, core::GgmlType::kQ5_1, d_mx, E * M, K,
                            nullptr);
        ok = hipGetLastError() == hipSuccess &&
             hipMemcpyAsync(h_iq4.data(), d_iq4, iq4_bytes,
                            hipMemcpyDeviceToHost, nullptr) == hipSuccess &&
             hipMemcpyAsync(h_mx.data(), d_mx, mx_bytes, hipMemcpyDeviceToHost,
                            nullptr) == hipSuccess &&
             hipDeviceSynchronize() == hipSuccess;
      }
      const auto run_format = [&](bool mxfp4, int tokens, bool raw,
                                  double& max_err) -> bool {
        max_err = 0.0;
        const int type_id = static_cast<int>(
            mxfp4 ? core::GgmlType::kMXFP4 : core::GgmlType::kIQ4_NL);
        const void* w_dev = mxfp4 ? d_mx : d_iq4;
        if (hipMemsetAsync(d_out, 0,
                           static_cast<std::size_t>(tokens) * M * sizeof(float),
                           nullptr) != hipSuccess)
          return false;
        const int rc =
            raw ? (mxfp4 ? qfn_mmq_mxfp4_moe_raw(w_dev, d_x, d_ids, d_out, M, K,
                                                 tokens, E, 1, nullptr)
                         : qfn_mmq_iq4_nl_moe_raw(w_dev, d_x, d_ids, d_out, M,
                                                  K, tokens, E, 1, nullptr))
                : qfn_mmq_moe_vec(type_id, w_dev, d_x, d_ids, d_out, M, K,
                                  tokens, E, 1, nullptr);
        if (rc != 0)
          return false;
        std::vector<float> h_out(static_cast<std::size_t>(tokens) * M);
        if (hipMemcpyAsync(h_out.data(), d_out, h_out.size() * sizeof(float),
                           hipMemcpyDeviceToHost, nullptr) != hipSuccess ||
            hipDeviceSynchronize() != hipSuccess)
          return false;
        for (int t = 0; t < tokens; ++t) {
          const int e = h_ids[t];
          for (int r = 0; r < M; ++r) {
            double ref = 0.0;
            for (int b = 0; b < B; ++b) {
              const int bidx = (e * M + r) * B + b;
              const float* xp = &h_x[static_cast<std::size_t>(t) * K + b * 32];
              if (mxfp4) {
                const auto& blk = h_mx[bidx];
                const double scale =
                    static_cast<double>(e8m0_to_f32(blk.e)) * 0.5;
                for (int j = 0; j < 16; ++j) {
                  const int q0 = blk.qs[j] & 0x0F;
                  const int q1 = blk.qs[j] >> 4;
                  ref += scale * kMxfp4Table[q0] * xp[j];
                  ref += scale * kMxfp4Table[q1] * xp[j + 16];
                }
              } else {
                const auto& blk = h_iq4[bidx];
                const double scale =
                    static_cast<double>(gufo::quant::Fp16ToFloat(blk.d));
                for (int j = 0; j < 16; ++j) {
                  const int q0 = blk.qs[j] & 0x0F;
                  const int q1 = blk.qs[j] >> 4;
                  ref += scale * gufo::quant::kValuesIq4Nl[q0] * xp[j];
                  ref += scale * gufo::quant::kValuesIq4Nl[q1] * xp[j + 16];
                }
              }
            }
            const float got = h_out[static_cast<std::size_t>(t) * M + r];
            if (!std::isfinite(got))
              return false;
            const double err =
                ref > got ? (ref - got) : (static_cast<double>(got) - ref);
            if (err > max_err)
              max_err = err;
          }
        }
        return true;
      };
      double e_iq2 = 0, e_iq24 = 0, e_iq40 = 0;
      double e_mx2 = 0, e_mx24 = 0, e_mx40 = 0;
      ok = ok && run_format(false, 2, false, e_iq2) &&
           run_format(false, 24, false, e_iq24) &&
           run_format(false, 40, true, e_iq40) &&
           run_format(true, 2, false, e_mx2) &&
           run_format(true, 24, false, e_mx24) &&
           run_format(true, 40, true, e_mx40);
      cleanup();
      std::fprintf(stderr,
                   "strix-moe-4bit-selftest: iq4_vec2=%.3e iq4_vec24=%.3e "
                   "iq4_mmq40=%.3e mx_vec2=%.3e mx_vec24=%.3e mx_mmq40=%.3e\n",
                   e_iq2, e_iq24, e_iq40, e_mx2, e_mx24, e_mx40);
      return ok && e_iq2 < 1e-4 && e_iq24 < 1e-4 && e_iq40 < 5e-3 &&
             e_mx2 < 1e-4 && e_mx24 < 1e-4 && e_mx40 < 5e-3;
    };
    if (!run_4bit_selftest()) {
      if (error_msg != nullptr)
        *error_msg = "MoE down 4-bit selftest failed";
      return nullptr;
    }
    if (!m->layers_.empty() &&
        m->layers_[0].ffn_gate_exps.type == core::GgmlType::kQ4_K &&
        m->layers_[0].ffn_up_exps.type == core::GgmlType::kQ4_K &&
        m->layers_[0].ffn_gate_exps.experts > 0) {
      const auto& g = m->layers_[0].ffn_gate_exps;
      const auto& u = m->layers_[0].ffn_up_exps;
      const int M = static_cast<int>(g.rows);
      const int K = static_cast<int>(g.cols);
      const int E = static_cast<int>(g.experts);
      constexpr int T = 16;
      const int U = static_cast<int>(m->config_.num_experts_used);
      if (K % 256 == 0 && U > 0 && U <= 32) {
        std::vector<float> h_x(static_cast<std::size_t>(T) * K);
        for (int t = 0; t < T; ++t) {
          for (int k = 0; k < K; ++k) {
            const int v = ((k * 17 + t * 29 + 3) % 97) - 48;
            h_x[static_cast<std::size_t>(t) * K + k] =
                static_cast<float>(v) * 0.015625f;
          }
        }
        std::vector<std::int32_t> h_ids(static_cast<std::size_t>(T) * U);
        for (int t = 0; t < T; ++t) {
          for (int j = 0; j < U; ++j) {
            if (j == 0 && t < 10)
              h_ids[static_cast<std::size_t>(t) * U + j] = 0;  // multi > 8
            else if (j == 1 && t < 5)
              h_ids[static_cast<std::size_t>(t) * U + j] = 3 % E;  // multi 5
            else if (j == 2 && t < 2)
              h_ids[static_cast<std::size_t>(t) * U + j] = 7 % E;  // multi 2
            else if (t == T - 1 && j == U - 1)
              h_ids[static_cast<std::size_t>(t) * U + j] = -1;     // sentinel
            else
              h_ids[static_cast<std::size_t>(t) * U + j] =
                  (17 + t * U + j) % E;  // singletons
          }
        }
        const std::size_t out_elems = static_cast<std::size_t>(T) * U * M;
        float* d_x = nullptr;
        std::int32_t* d_ids = nullptr;
        float* d_ref = nullptr;
        float* d_fast = nullptr;
        const auto free_gu = [&]() {
          if (d_x) (void)hipFree(d_x);
          if (d_ids) (void)hipFree(d_ids);
          if (d_ref) (void)hipFree(d_ref);
          if (d_fast) (void)hipFree(d_fast);
        };
        bool gu_ok =
            hip::AllocateDevice(&d_x, h_x.size() * sizeof(float)) == hipSuccess &&
            hip::AllocateDevice(&d_ids, h_ids.size() * sizeof(std::int32_t)) ==
                hipSuccess &&
            hip::AllocateDevice(&d_ref, out_elems * sizeof(float)) ==
                hipSuccess &&
            hip::AllocateDevice(&d_fast, out_elems * sizeof(float)) ==
                hipSuccess &&
            hipMemcpyAsync(d_x, h_x.data(), h_x.size() * sizeof(float),
                           hipMemcpyHostToDevice, nullptr) == hipSuccess &&
            hipMemcpyAsync(d_ids, h_ids.data(),
                           h_ids.size() * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, nullptr) == hipSuccess &&
            qfn_mmq_moe_gu_selftest_compare(g.data, u.data, d_x, d_ids, d_ref,
                                            d_fast, M, K, T, U, nullptr) == 0;
        std::vector<float> h_ref(out_elems), h_fast(out_elems);
        gu_ok = gu_ok &&
                hipMemcpyAsync(h_ref.data(), d_ref, out_elems * sizeof(float),
                               hipMemcpyDeviceToHost, nullptr) == hipSuccess &&
                hipMemcpyAsync(h_fast.data(), d_fast, out_elems * sizeof(float),
                               hipMemcpyDeviceToHost, nullptr) == hipSuccess &&
                hipDeviceSynchronize() == hipSuccess;
        free_gu();
        double max_err = 0.0;
        bool all_finite = true;
        if (gu_ok) {
          for (std::size_t i = 0; i < out_elems; ++i) {
            if (!std::isfinite(h_ref[i]) || !std::isfinite(h_fast[i])) {
              all_finite = false;
              continue;
            }
            const double d = std::abs(static_cast<double>(h_ref[i]) -
                                      static_cast<double>(h_fast[i]));
            if (d > max_err)
              max_err = d;
          }
        }
        std::fprintf(stderr, "strix-moe-gu-selftest: tokens=%d max_err=%.3e\n",
                     T, max_err);
        if (!gu_ok || !all_finite || max_err >= 1e-5) {
          if (error_msg != nullptr)
            *error_msg = "MoE Wave32 gate/up selftest failed";
          return nullptr;
        }
      }
    }
    const DeviceTensor* probe = nullptr;
    const DeviceTensor* hc_probe = nullptr;
    const auto check_probe = [&](const DeviceTensor& t) {
      if (probe == nullptr && t.q4k != nullptr && t.cols % 256 == 0)
        probe = &t;
    };
    if (m->has_mtp_) {
      check_probe(m->mtp_.attn_qkv);
      check_probe(m->mtp_.attn_out);
      check_probe(m->mtp_.nextn_fc_embedding);
    }
    if (!m->layers_.empty()) {
      check_probe(m->layers_[0].ssm_in);
      check_probe(m->layers_[0].ssm_out);
      check_probe(m->layers_[0].attn_qkv);
      if (m->layers_[0].hc_attn.down.q4k != nullptr &&
          m->layers_[0].hc_attn.down.cols % 256 == 0) {
        hc_probe = &m->layers_[0].hc_attn.down;
      } else if (m->has_mtp_ && m->mtp_.hc_attn.down.q4k != nullptr &&
                 m->mtp_.hc_attn.down.cols % 256 == 0) {
        hc_probe = &m->mtp_.hc_attn.down;
      }
    }
    if (probe != nullptr &&
        qfn_mmq_q4_K_wmma_selftest(probe->q4k, static_cast<int>(probe->rows),
                                   static_cast<int>(probe->cols),
                                   nullptr) != 0) {
      if (error_msg != nullptr)
        *error_msg = "Q4_K WMMA selftest failed";
      return nullptr;
    }
    if (hc_probe != nullptr &&
        qfn_mmq_q4_K_wmma_selftest(hc_probe->q4k,
                                   static_cast<int>(hc_probe->rows),
                                   static_cast<int>(hc_probe->cols),
                                   nullptr) != 0) {
      if (error_msg != nullptr)
        *error_msg = "Q4_K HC down split-K WMMA selftest failed";
      return nullptr;
    }
  }
  if (HwBenchEnabled()) {
    hipEvent_t e0 = nullptr, e1 = nullptr;
    if (hipEventCreate(&e0) == hipSuccess &&
        hipEventCreate(&e1) == hipSuccess) {
      // 1) 128 MiB D2D copy bandwidth (reads + writes = 256 MiB per copy)
      constexpr std::size_t kCopyBytes = std::size_t{128} << 20;
      void* d_src = nullptr;
      void* d_dst = nullptr;
      if (hip::AllocateDevice(&d_src, kCopyBytes) == hipSuccess &&
          hip::AllocateDevice(&d_dst, kCopyBytes) == hipSuccess) {
        (void)hipMemsetAsync(d_src, 0x5A, kCopyBytes, nullptr);
        (void)hipMemsetAsync(d_dst, 0x00, kCopyBytes, nullptr);
        (void)hipDeviceSynchronize();
        for (int w = 0; w < 3; ++w)
          (void)hipMemcpyAsync(d_dst, d_src, kCopyBytes,
                               hipMemcpyDeviceToDevice, nullptr);
        (void)hipDeviceSynchronize();
        float t[10]{};
        bool ok = true;
        for (int i = 0; i < 10; ++i) {
          ok = ok && hipEventRecord(e0, nullptr) == hipSuccess &&
               hipMemcpyAsync(d_dst, d_src, kCopyBytes,
                              hipMemcpyDeviceToDevice, nullptr) == hipSuccess &&
               hipEventRecord(e1, nullptr) == hipSuccess &&
               hipEventSynchronize(e1) == hipSuccess &&
               hipEventElapsedTime(&t[i], e0, e1) == hipSuccess;
        }
        if (ok) {
          std::sort(t, t + 10);
          const double med = 0.5 * (t[4] + t[5]);
          const double gbs =
              med > 0.0 ? (2.0 * static_cast<double>(kCopyBytes)) /
                              (med * 1e-3) / 1e9
                        : 0.0;
          std::fprintf(stderr,
                       "strix-hw-bench: copy_128MiB_ms=%.3f copy_gbs=%.1f\n",
                       med, gbs);
        }
      } else {
        (void)hipGetLastError();
        std::fprintf(stderr, "strix-hw-bench: copy alloc skipped\n");
      }
      if (d_src) (void)hipFree(d_src);
      if (d_dst) (void)hipFree(d_dst);

      // 2) FP16 square GEMM throughput (4096^3 and 8192^3)
      hipblasHandle_t blas = nullptr;
      if (hipblasCreate(&blas) == HIPBLAS_STATUS_SUCCESS) {
        for (int N : {4096, 8192}) {
          const std::size_t bytes = std::size_t{N} * N * 2;
          void* dA = nullptr;
          void* dB = nullptr;
          void* dC = nullptr;
          if (hip::AllocateDevice(&dA, bytes) == hipSuccess &&
              hip::AllocateDevice(&dB, bytes) == hipSuccess &&
              hip::AllocateDevice(&dC, bytes) == hipSuccess) {
            (void)hipMemsetAsync(dA, 0, bytes, nullptr);
            (void)hipMemsetAsync(dB, 0, bytes, nullptr);
            (void)hipMemsetAsync(dC, 0, bytes, nullptr);
            (void)hipDeviceSynchronize();
            const float alpha = 1.0f;
            const float beta = 0.0f;
            bool ok = true;
            for (int w = 0; w < 2; ++w) {
              ok = ok &&
                   hipblasGemmEx(blas, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N,
                                 &alpha, dA, HIP_R_16F, N, dB, HIP_R_16F, N,
                                 &beta, dC, HIP_R_16F, N, HIPBLAS_COMPUTE_32F,
                                 HIPBLAS_GEMM_DEFAULT) ==
                       HIPBLAS_STATUS_SUCCESS;
            }
            (void)hipDeviceSynchronize();
            float t[5]{};
            for (int i = 0; i < 5 && ok; ++i) {
              ok = ok && hipEventRecord(e0, nullptr) == hipSuccess &&
                   hipblasGemmEx(blas, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N,
                                 &alpha, dA, HIP_R_16F, N, dB, HIP_R_16F, N,
                                 &beta, dC, HIP_R_16F, N, HIPBLAS_COMPUTE_32F,
                                 HIPBLAS_GEMM_DEFAULT) ==
                       HIPBLAS_STATUS_SUCCESS &&
                   hipEventRecord(e1, nullptr) == hipSuccess &&
                   hipEventSynchronize(e1) == hipSuccess &&
                   hipEventElapsedTime(&t[i], e0, e1) == hipSuccess;
            }
            if (ok) {
              std::sort(t, t + 5);
              const double med = static_cast<double>(t[2]);
              const double flops = 2.0 * N * N * N;
              const double tflops =
                  med > 0.0 ? flops / (med * 1e-3) / 1e12 : 0.0;
              std::fprintf(stderr,
                           "strix-hw-bench: gemm_fp16_N=%d ms=%.3f tflops=%.2f\n",
                           N, med, tflops);
            }
          } else {
            (void)hipGetLastError();
            std::fprintf(stderr, "strix-hw-bench: gemm_fp16_N=%d alloc skipped\n",
                         N);
          }
          if (dA) (void)hipFree(dA);
          if (dB) (void)hipFree(dB);
          if (dC) (void)hipFree(dC);
        }
        (void)hipblasDestroy(blas);
      }
    }
    if (e0) (void)hipEventDestroy(e0);
    if (e1) (void)hipEventDestroy(e1);

    // 3) Realistic correlated MoE gate+up Q4_K wave64 vs wave32 microbench
    if (!m->layers_.empty() &&
        m->layers_[0].ffn_gate_exps.type == core::GgmlType::kQ4_K &&
        m->layers_[0].ffn_up_exps.type == core::GgmlType::kQ4_K &&
        m->layers_[0].ffn_gate_exps.experts > 0) {
      const auto& g = m->layers_[0].ffn_gate_exps;
      const auto& u = m->layers_[0].ffn_up_exps;
      const int M = static_cast<int>(g.rows);
      const int K = static_cast<int>(g.cols);
      const int E = static_cast<int>(g.experts);
      const int U = static_cast<int>(m->config_.num_experts_used);
      if (K % 256 == 0 && U > 0 && U <= 32) {
        for (int T : {12, 16}) {
          std::vector<float> h_x(std::size_t{T} * K);
          for (std::size_t i = 0; i < h_x.size(); ++i)
            h_x[i] = static_cast<float>((i % 37) - 18) * 0.015625f;
          std::vector<std::int32_t> h_ids(std::size_t{T} * U);
          std::vector<std::uint8_t> seen(E, 0);
          int uniq = 0;
          const int pool_span = std::min(E, 72);
          for (int t = 0; t < T; ++t) {
            for (int j = 0; j < U; ++j) {
              const int e =
                  (((t * 7 + j * 13) % pool_span) + (t / 4) * 18) % E;
              h_ids[std::size_t{t} * U + j] = e;
              if (!seen[e]) {
                seen[e] = 1;
                ++uniq;
              }
            }
          }
          float* d_x = nullptr;
          std::int32_t* d_ids = nullptr;
          float* d_out = nullptr;
          const std::size_t out_elems = std::size_t{T} * U * M;
          if (hip::AllocateDevice(&d_x, h_x.size() * sizeof(float)) ==
                  hipSuccess &&
              hip::AllocateDevice(&d_ids, h_ids.size() * sizeof(std::int32_t)) ==
                  hipSuccess &&
              hip::AllocateDevice(&d_out, out_elems * sizeof(float)) ==
                  hipSuccess) {
            float w64_ms = 0.0f, w32_ms = 0.0f;
            if (hipMemcpyAsync(d_x, h_x.data(), h_x.size() * sizeof(float),
                               hipMemcpyHostToDevice, nullptr) == hipSuccess &&
                hipMemcpyAsync(d_ids, h_ids.data(),
                               h_ids.size() * sizeof(std::int32_t),
                               hipMemcpyHostToDevice, nullptr) == hipSuccess &&
                hipDeviceSynchronize() == hipSuccess &&
                qfn_mmq_moe_gu_bench(g.data, u.data, d_x, d_ids, d_out, M, K, T,
                                     U, &w64_ms, &w32_ms, nullptr) == 0) {
              const double spd = w32_ms > 0.0f ? w64_ms / w32_ms : 0.0;
              std::fprintf(
                  stderr,
                  "strix-hw-bench: moe_gu_T=%d uniq=%d w64_ms=%.3f "
                  "w32_ms=%.3f speedup=%.3fx\n",
                  T, uniq, w64_ms, w32_ms, spd);
            }
          } else {
            (void)hipGetLastError();
          }
          if (d_x) (void)hipFree(d_x);
          if (d_ids) (void)hipFree(d_ids);
          if (d_out) (void)hipFree(d_out);
        }
      }
    }
  }
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
