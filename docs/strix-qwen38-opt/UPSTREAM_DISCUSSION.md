# Qwen3.8-Flash-Next Strix Halo (`gfx1151`) Optimization Findings: +47.8% Constrained Tool-Calling & +29.7% 3-Stream Decode vs. `v0.35.0`

Hi Gufo team & community!

We ran a 15-window hardware-ceiling optimization study on **AMD Strix Halo (Radeon 8060S, `gfx1151`, 122 GiB unified LPDDR5X)** for **`Qwen3.8-Flash-Next-UD-Q4_K_XL` + MTP** (`--sessions 3 --context 87040 --speculative mtp`), benchmarking head-to-head against both **stock `v0.24.0`** and **stock `v0.35.0`** (`0260e11`).

Full branch, reproducible Nix flake tag, launch script, and 37-arm structured ledger:
**`https://github.com/poitee/gufo/tree/strix-qwen38-opt`** (pinned tag: `v0.24.0-strix-w15`)

---

## Measured Hardware Ceilings (`gfx1151`)
Using a direct HIP microbenchmark (`GUFO_STRIX_HW_BENCH=1`) on the same machine:
- **Device copy BW (`128 MiB` `hipMemcpyDeviceToDevice`):** `214.6 GB/s` (`1.251 ms`)
- **FP16 GEMM (`hipblasGemmEx`):** `33.43 TFLOPS` (`4096³`) / `25.38 TFLOPS` sustained (`8192³`)
- **Empirical unique routed experts at `C=2` (`avg_rows = 13.0`):** `69.5 / 512` (`13.6%`, vs `116/512` i.i.d.)

---

## Head-to-Head Summary (Same Machine, Same Model, Same Serving Contract)

| Metric | Stock `v0.24.0` | Stock `v0.35.0` | Pure Stock Weights (`v15-ll-w15`) | **MoE-Lossless (`v15-ml-w15`)** | `v15-ml-w15` vs. `v0.35.0` |
|---|---:|---:|---:|---:|---:|
| **Single-stream (`C1` greedy)** | `61.8 tok/s` | `71.0 tok/s` | `71.9 tok/s` | **`77.1 tok/s`** *(93–96% of 4K ceil)* | **`+8.6%`** |
| **3-stream greedy (`C3g`)** | `72.4 tok/s` | `79.1 tok/s` | `89.5 tok/s` | **`94.5 tok/s`** | **`+19.5%`** |
| **3-stream sampled (`C3s`, `T=0.7`)** | `74.3 tok/s` | `77.5 tok/s` | `88.2 tok/s` | **`100.5 tok/s`** | **`+29.7%`** |
| **Pi coding (`434`-case C=2)** | `43.08` (`111.0 ms/r`) | `40.50` (`140.3 ms/r`) | `55.3` *(40-slice)* | **`59.86` (`96.0 ms/r`)** | **`+47.8%`** |
| **omp agent (`434`-case C=2)** | `31.70` (`86.5 ms/r`) | `31.10` (`106.4 ms/r`) | `40.6` *(40-slice)* | **`43.06` (`88.9 ms/r`)** | **`+38.5%`** |
| **Legacy20 (`434`-case C=2)** | `54.40` (`20/20`) | `48.72` (`18/20`) | — | **`57.50` (`20/20`)** | **`+18.0%`** |
| **Full `434` pass & hard fails** | `403/434` (`hf=1`) | `384/434` (`hf=1`) | `38/40` slice (`hf=0`) | **`408/434` (`94.0%`, `hf=0`)** | **`+24` passes, `0` hard fails** |
| **Full `434` wall time** | `847s` | `942s` | — | **`603s`** | **`1.56×` faster** |
| **Cold needles (`50%` / `90%`)** | — | `3/3 @ 1,635` · `3/3 @ 1,593` | — | **`3/3 @ 1,615` · `3/3 @ 1,598`** | **Parity (`~1,600 tok/s`)** |

---

## 4 Standalone Lossless Wins Worth Porting to `main` (`v0.35.0+`)

Upstream `v0.25.0 → v0.35.0` already landed `Q4_K` draft heads (`#1209`), confidence-gated MTP (`#882`), and fused HC/prefill work. However, four lossless wins from our branch are still not in `v0.35.0` and account for the large gap on constrained tool-calling (`Pi 59.86` vs `40.50 tok/s`, `408/434` vs `384/434`) and `C=3` parallel decode (`100.5` vs `77.5 tok/s`):

1. **Cached GPU Grammar Bitmask (`512`-entry LRU) + `ProseLexeme` in `WithTools`:**
   - Uncached grammar masking spiked candidate-prep latency (`0.5 → 10.5 ms/round`), while caching bitmasks in a 512-entry LRU and applying them on-GPU brought `cand` back to `0.5 ms/round` and drove the constrained tool-call speedup (`Pi 140.3 → 96.0 ms/r`).
   - Adding `ProseLexeme` in `WithTools` eliminated the `omp/hostile_result/007` hard fail (`hf=0` across all 434-case runs).
2. **Batched Piecewise HIP Graphs (`GUFO_STRIX_BATCH_GRAPH=1`):**
   - Capturing the steady-state decode layers eliminated launch overhead across both `C=2` and `C=3` batches (`6/6` exact greedy `C1` SHA match).
3. **True Wave32 Compilation for Split `moe_gu` (`moe_wave32.hip.cpp`):**
   - In `CMakeLists.txt`, `moe_wave64.hip.cpp` is compiled with `-mwavefrontsize64`. Extracting the wave32 split kernels into a separate `moe_wave32.hip.cpp` (default `wf=32`) yielded **`1.058×` (`T=12`) / `1.082×` (`T=16`)** on `moe_gu` with **`max_err = 0.000e+00`** (pushing `moe_gu` to ~`91.4%` of copy BW).
4. **Packed-K16 `Q8_0` WMMA for Short-K Dense Projections (`chunk <= 24`):**
   - `mul_mat_q8_decode_batch` uses 4 half-zero-padded WMMAs per `Q8_0` block, `181` VGPRs, `32 KiB` LDS (25% occupancy), and 8 fixed waves (leaving 6/8 waves idle on `K=320` `hc_up`!).
   - Packing full K=16 (2 WMMAs/block, single FP32 accumulator, `3–5 KiB` LDS) with adaptive `2/4` waves for `chunk <= 24` gave **`1.577×` on `hc_up` (`10240×320`)** and **`1.441×` on `shexp_down` (`2560×640`)** (`max_err ≤ 7.15e-07`), while keeping `lm_head` (`K=2560`) on the existing `212 GB/s` 8-wave kernel.

Happy to break any of these 4 into clean, focused PRs against `main` if there's interest!
