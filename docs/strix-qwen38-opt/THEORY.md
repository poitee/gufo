# Theoretical Ceilings & Kernel Audit — `Qwen3.8-Flash-Next` (`qwen38-gufo`) on Strix Halo (`gfx1151`)

Methodology: [`inference-optimisation/references/ceilings.md`] and [`scripts/ceiling.py`].

> [!NOTE]
> **Hardware Status: MEASURED (`GUFO_STRIX_HW_BENCH=1`, Window 13).**
> Replaced earlier `240 GB/s` / `59 TFLOPS` placeholders with direct HIP measurements on Strix Halo (`gfx1151`) ([`hwbench-v13.txt`]).

---

## 1. Measured Hardware & Model Geometry

- **Host / Accelerator:** Strix Halo (`gfx1151`) — AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`), `122 GiB` unified LPDDR5X (`amdgpu.gttsize=126976`, `ttm.pages_limit=32505856`), Kingston `OM8TAP42048K1-A00` PCIe 4.0 NVMe.
- **Direct HIP Hardware Microbenchmark ([`hwbench-v13.txt`]):**
  - **Device copy bandwidth (`128 MiB` `hipMemcpyDeviceToDevice` median):** **`214.6 GB/s`** (`1.251 ms`).
  - **FP16 square GEMM (`hipblasGemmEx`, `HIPBLAS_COMPUTE_32F`):** **`33.43 TFLOPS`** at `4096³` (`4.112 ms`), dropping to **`25.38 TFLOPS`** sustained at `8192³` (`43.314 ms`).
- **Real Host-PID Unified Memory Footprint (`drm-resident-gtt/vram` + `VmRSS`):**
  - Stock `qwen38-gufo` (`0.24.0`): **`92,170 MiB` DRM** (`92,165 GTT` + `4 VRAM`) + `2,704 MiB` RSS = **`94,874 MiB`** startup total.
  - Window 13 winner `v13-iq4all`: **`86,024 MiB` DRM** + `2,663 MiB` RSS = **`88,688 MiB`** startup total (**`-6,146 MiB` / `-6.0 GiB` net DRM savings** after `Q4K_DENSE`/`Q4K_MTP`/`Q4K_DRAFT` copies); stayed at **`86,984 MiB` DRM** (`98,817 MiB` total with `11.8 GiB` host cache RSS) after 434 evals + 78K-token needle gate.
- **Serving Contract & Architecture ([`config.hpp`]):**
  - `--sessions 3 --context 87040 --reasoning-budget 2048 --cache-ram-bytes 12884901888 --cache-disk /cache --cache-disk-bytes 34359738368 --stall-timeout-ms 120000 --speculative mtp` (`draft_limit = 7`).
  - `48` trunk layers (`36` GDN linear-attention + `12` QSA full-attention, `full_attention_interval = 4`), `512` routed experts (`top_k = 10`, `expert_ff = 640`) + `1` shared expert, `hidden_size = 2560`, `hc_count = 4`, `vocab_size = 248320`, `indexer_top_k = 2048`, `compress_ratio = 4`.

---

## 2. Exact GGUF Checkpoint Byte Breakdown

| Component | Raw GGUF Size | Active Decode Behavior (`v12-p2-p1` vs `v13-iq4all`) |
|---|---:|---|
| `moe_gate_up_exps` (`Q4_K` 44.355 + `Q5_K` 1.153 GB) | **45.508 GB** | Empirical **`69.8 / 512` (`13.6%`)** unique experts at `avg_rows=13.0–13.2` (≈ `6.08–6.20 GB/fwd`, vs `22.6%` i.i.d.) |
| `moe_down_exps` (`Q5_1` 27.053 + `Q8_0` 4.456 GB) | **31.510 GB** | `v12-p2-p1`: `31.510 GB` (`~4.30 GB/fwd`); `v13-iq4all` (`48L IQ4_NL`, `-8450 MiB`): **`22.650 GB`** (`~3.03 GB/fwd`, total experts `68.158 GB`) |
| `tok_embd` (PLE n-gram lookup table) | **29.476 GB** | Lookup-only per token (~`0` streaming decode bytes/token; excluded from decode BW model) |
| Dense trunk (`attn` 2.623 + `ssm` 0.643 + `hc` 0.351 + `rtr` 0.252 + `shexp` 0.251 + `conv` 0.035) | **4.155 GB** | **`2.585 GB`** effective (`288` `Q4_K` copies = `1687 MiB` replacing ~`3.34 GB` `Q8_0`) |
| `lm_head` (`output.weight`, `248,320 × 2560` `Q8_0`) | **0.675 GB** | Read once per verify forward (`3.2 ms` = `210.9 GB/s` = **`98.3%`** of copy ceiling) |
| MTP sidecar (`2.776 GB` raw + draft head) | **2.776 GB** | `Q4K_MTP` (`940 MiB`) + `MTP_DOWN_IQ4` (`-400 MiB`) -> `draft_layer ≈ 0.060–0.068 GB`; `98,304`-row `Q4_K` draft head = `0.142 GB` (`135 MiB`) |

---

## 3. KV Cache & GDN Recurrent State Budget

Exact per-stream layout from [`continuation_layout.cpp`]:

- **Full-attention KV + pooled indexer keys:** `12` trunk + `1` MTP layers × `(1024 B K + 1024 B V + 64 B amortized pooled key)` = **`27,456 B/token`** (`2.39 GB` per `87,040`-token stream, **`7.17 GB`** for 3 full streams, `7.20 GB` per `262K` stream). QSA (`indexer_top_k = 2048`) caps decode KV gather at `≤ 54.5 MB` + pooled indexer keys (`832 B/tok`).
- **Fixed recurrent + conv + ring state:** `36` GDN recurrent matrices (`113.25 MB`, `~0.227 GB` R+W/stream/round) + `36` conv buffers (`4.42 MB`) + QSA raw ring (`13.63 MB`) + PLE/logits (`1.36 MB`) ≈ **`132.7 MB/stream`** (`~0.40 GB` for 3 streams).

---

## 4. Measured `ceiling.py` Tables (`214.6 GB/s`, `33.43 TFLOPS`, `MFU = 0.5`)

Configs: [`ceiling-stock.json`], [`ceiling-qwen38-Strix Halo host.json`] (`v12-p2-p1` control), [`ceiling-v13-iq4all.json`] (Window 13 winner), [`ceiling-v13-target.json`] (future pruned draft head target).

| Config | C1 Ceiling @ 4K (`drafts=3..7`) | C1 Achieved | C2 Ceiling @ 4K | C2 Achieved (`Pi` / `omp` agg.) | C3 Ceiling @ 4K | C3 Achieved (`C3g` / `C3s`) |
|---|---:|---:|---:|---:|---:|---:|
| **Stock `0.24.0` (`base-b`)** | `59–61` tok/s | `61.8` (`~101–105%`*) | `137–153` tok/s | `Pi 86.2` (`~56–63%`), `omp 63.4` (`~41–46%`) | `225–254` tok/s | `72.4` / `74.3` (`~29–33%`) |
| **`v12-p2-p1` (`v13-ctrl`)** | `80–83` tok/s | `77.3` (`93–97%`) | `189–204` tok/s | `Pi 118.1` (`~58–62%`), `omp 83.3` (`~41–44%`) | `313–338` tok/s | `94.6` / `96.4` (`~28–31%`) |
| **`v13-iq4all` (Win 13)** | **`87–89` tok/s** | **`80.2`** *(peak 83.9)* (`90–92%`) | **`208–223` tok/s** | **`Pi 124.0` (`~56–60%`), `omp 88.8` (`~40–43%`)** | **`345–371` tok/s** | **`100.2` / `103.3` (`~27–30%`)** |
| **Future `+ 32K` fast draft** | `92–93` tok/s | — | `220–232` tok/s | — | `367–388` tok/s | — |

*\*C1 vs `ceiling.py` exceeds 100% slightly because `ceiling.py` assumes i.i.d. uniform expert sampling (`1-(1-k/E)^M`), whereas empirical speculative batches exhibit strong expert reuse (`69.8/512` at `avg_rows=13.0`, vs `116/512` i.i.d.).*

- **Prefill ceiling vs achieved:** `ceiling.py` gives `1274 / 963 / 659 tok/s` (`4K / 32K / 85K`) assuming dense FP16 attention across all 12 full-attn layers (`0.5 × 33.43 TFLOPS`). In practice, QSA (`top_k = 2048`) caps long-context attention FLOPs and `v13-ctrl` achieved **`1,636.3 tok/s`** at `48,841` tokens (~61% of full FP16 GEMM peak on linear ops). `v13-iq4all` (`IQ4_NL` `moe_down`) prefilled at **`1,101.9` (`44.8K`) and `1,102.6 tok/s` (`78.2K`)** — flat across context, ~33% slower prefill than `Q5_1/Q8_0` due to `IQ4_NL` dequant cost in compute-bound GEMMs (labelled trade-off).

---

## 5. Per-Kernel Forward Breakdown, Unique Experts & ISA Audit (`gfx1151`)

Measured via `GUFO_STRIX_PROFILE_FWD=1` ([`prof-v13-fwd.txt`]) and AMDGPU `.note` metadata ([`isa-v13.txt`]):

| Stage / Kernel | Time (`ctrl` vs `iq4all`) | Real Bytes Read (`uniq=69.8/68.4`) | Effective BW (`% of 214.6 GB/s`) | ISA Metadata (`isa-v13.txt`) & Diagnosis |
|---|---:|---:|---:|---|
| **`lm_head` (`Q8_0`)** | `3.2 ms` / `3.2 ms` | `0.675 GB` | **`210.9 GB/s` (`98.3%`)** | Near hardware copy ceiling; no actionable decode BW headroom left. |
| **`moe_dn`** | `21.5 ms` → **`16.6 ms`** | `4.30 GB` → **`3.03 GB`** | `199.8` (`93.1%`) → `182.3 GB/s` (`84.9%`) | Native `wf=32`, `scratch=0`, `vgpr=36..83` (`Q5_1`), `38` (`IQ4_NL`). `IQ4_NL` saved `-22.8%` `moe_dn` ms and `-6.0 GiB` DRM. |
| **`moe_gu` (`Q4_K`)** | `32.4 ms` / `31.9 ms` | `6.20 GB` / `6.08 GB` | **`191.5 GB/s` (`89.2%`)** | **Explains `v13-fastgu` neutrality:** already at ~89% of `214.6 GB/s` copy ceiling! **ISA bug found:** `moe_wave64.hip.cpp` has `-mwavefrontsize64` in `CMakeLists.txt`, forcing `_w32` kernels to compile as **`wf=64` (`vgpr=116` multi, `82` single, 32 lanes dead)**. |
| **`gdn` (`36L`)** | `3.5 ms` / `3.6 ms` | `~0.454 GB` (2-stream R+W) | `129.7 GB/s` (`60.4%`) | Recurrent state R+W + conv across 36 linear layers. |
| **Dense (`hc+qkv+out+rtr+shexp`)** | **`20.9 ms` / `21.4 ms`** | `~2.58 GB` (`Q4K` + `Q8`) | **`123.7 GB/s` (`57.6%`)** | **Largest remaining BW gap in fwd (`+9–10 ms` opportunity)!** ISA shows `mul_mat_q4k_decode_batch<..., has_gate=true>` spills **`scratch=12`** (`vgpr=94..176`) and `q8_decode` uses `vgpr=110..181`, `lds=32768`. |
| **`mtp_heads` (`Q4_K`)** | `9.6 ms` / `9.0 ms` | `~0.94 GB` (`6.6 × 0.142 GB`) | **`97.6 GB/s` (`45.5%`)** | **Explains `dv48`/`dv32` failure:** `dv48` dropped small-head hit rate `sh=48/48 → 19/48` (falling back to 248K `Q8_0` head, `mtp_heads 9.6 → 22.1 ms`). Also `hwbench` showed `q4k_draft` ran `71–73 GB/s` at `M=1` or `<98K` rows vs **`232.1 GB/s`** (`0.610 ms`) at `98304 M=2` once GPU memory/IF clocks boosted cleanly. |
