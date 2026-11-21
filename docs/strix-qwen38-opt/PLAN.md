# Qwen3.8-Flash-Next (`qwen38-gufo`) Optimisation Plan — Strix Halo (`gfx1151`)

Methodology: [inference-optimisation] (`fda968e7fc1996aa23c67139c755fe541c9d4be0`).
Ceilings & Kernel Audit: [THEORY.md] | Structured Ledger: [ledger.jsonl]

## Targets & Constraints

- **Target hardware (Strix Halo (`gfx1151`)):** AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`), `122 GiB` unified LPDDR5X (**measured `214.6 GB/s` copy BW, `33.43 / 25.38 TFLOPS` FP16**).
- **Build host (`build-host`, Linux build VM):** `10c/32G` (`net0 queues=4`). **Always use `nix build --cores 8` / `--parallel 8`.**
- **Target model & runtime:** `Qwen3.8-Flash-Next-UD-Q4_K_XL` (`111.324 GB`) + `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (`2.776 GB`), branch `q4k-dense` in [work].
- **Serving contract:** `--sessions 3 --context 87040 --reasoning-budget 2048 --cache-ram-bytes 12884901888 --cache-disk /cache --cache-disk-bytes 34359738368 --stall-timeout-ms 120000 --speculative mtp` (`guarded-run 92`).
- **Primary metrics & equivalence rules:** `C1` (`>=6` samples median), `C3g`/`C3s` (`3×3`), `toolcall_eval.py` (`Pi`, `omp`, `leg`, `40-slice` + `434-case`, `hard_fail == 0`, `hr7=pass`, zero `[ERROR]`), host-PID DRM residency + `VmRSS`, and `50%`/`90%` long-context needle retrieval ([needle_gate.py]).

---

## Baseline & Historical Ledger (Windows 1–13)

| Window / Arm | Key Change / Flags | C1 (tok/s) | C3g / C3s (tok/s) | 40-Slice | 434-Case Full Suite | Verdict & Why |
|---|---|---:|---:|---|---|---|
| **Stock `0.24.0` (`base-b`)** | Production baseline (`drm=92,170MiB`, `total=94,874MiB`) | 61.8 | 72.4 / 74.3 | `37/40` (`hf=0`, Pi `43.1`, omp `29.8`) | `403/434` (`hf=1`, `847s`), Pi `43.08`, omp `31.70`, leg `54.40` | Baseline reference (`base-b`); `hf=1` on `omp/hostile_result/007`. |
| **Window 8 (`0b58045`)** | RDNA3.5 Q4_K WMMA + Q4K MTP + wide `Q4K_DENSE` | 77.2 | 91.8 / 99.3 | `39/40` (`hf=0`) | `400/434` (`hf=0`, `685s`), Pi `48.98`, omp `36.81`, leg `62.01` | First wide Q4K dense win (`+13.7%` Pi, `+16.1%` omp, `hf=0`). |
| **Window 10 (`bc0d602`)** | Fast constrained verification + `SPLITK_Q8=1` | 73.7 (`lossl`) / 79.2 (`wide`) | 88.2 / 86.1 (`lossl`)<br>94.8 / 97.3 (`wide`) | `wide`: `39/40` (Pi `59.3`, omp `45.3`) | `v10-lossl`: `401/434` (`697s`), Pi `53.79`, omp `38.87`<br>`v10b-wide`: `402/434` (`594s`, `hf=1`) | Major constrained-decoding speedup (`Pi +24.8%` lossless, `+37.4%` wide). |
| **Window 12 (`4fa37f7`)** | `ProseLexeme` in `WithTools`, cached P1 `GPU_GRAMMAR_MASK=1` (512 LRU), isolated `BATCH_GRAPH=1` ([verdict-v12.txt]) | 76.6 | 96.5 / 98.6 | `38/40` (`hf=0`, Pi `61.3`, omp `45.2`) | **`v12-win` (`v12-p2-p1`):** `398/434` (**`hf=0`, `hr7=pass`, `582s`**), Pi **`58.68`** (`+36.2%`), omp **`43.22`** (`+36.3%`), leg `63.81` (`19/20`) | Cured v11's `omp` regression (`cand=0.5ms`), eliminated `hr7`, fastest 434 wall time (`582s`, 1.46× faster), bit-exact `T=0 C=1`. |
| **Window 13 Controls (`075d504`, `ctrl-a/b/c`)** | Interleaved `v12-p2-p1` controls (`n=6` C1 warmup-discarded) + `GUFO_STRIX_HW_BENCH=1` + `uniq` profiler | `77.2` / `77.4` / `77.3` | `97.5/96.6` · `94.8/96.4` · `94.6/96.4` | `38/40` · `36/40` · `39/40` (`hf=0`) | **`v13-ctrl`:** `400/434` (`hf=0`, `657s`), Pi `59.07` (`+37.1%`), omp `41.63` (`+31.3%`), leg **`63.20`** (**`20/20`**) | Tight control repeatability (`C1 77.2–77.4`); `50pct` cold needle **`48.8K`: `3/3` @ `1636.3 tok/s`**. |
| **Window 13 `v13-dv48` & `v13-dv32`** | `GUFO_STRIX_DRAFT_VOCAB=49152` / `32768` | 75.1 / 72.5 | 92.7/94.9 · 89.2/92.5 | `39/40` / `38/40` | Skipped (rejected) | **REJECTED:** `T=0 C=1` bit-exact (`6/6`), but `sh` dropped `48/48 -> 19/48` (`mtp_heads 9.6 -> 22.1ms`), regressing `C1/C3/Pi/omp`. |
| **Window 13 `v13-fastgu`** | Isolated `GUFO_STRIX_FAST_MOE_GU=1` (`PROFILE_FWD=1`) | 75.8 | 94.4 / 96.5 | `38/40` (`hf=0`) | Skipped (rejected) | **REJECTED:** Neutral (`gm=65.4`). `moe_gu=32.4ms` already runs at `89.2%` of `214.6 GB/s`, and `CMakeLists.txt` forced `_w32` into `wf=64` (`vgpr=116`). |
| **Window 13 Winner: `v13-iq4all` (`075d504`)** | `v12-p2-p1` + `MOE_DOWN=iq4_nl` (`48L`, `-8450 MiB`) + `MTP_DOWN_IQ4=1` (`-400 MiB`) ([verdict-v13.txt]) | **`80.2`** (`+29.8%` vs base) | **`100.2` / `103.3`** (`+38.4%` / `+39.0%`) | `38/40` (`hf=0`, Pi **`65.2`**, omp **`46.3`**, `gm=70.2`) | **`399/434`** (`hf=0`, `hr7=pass`, `612s`), Pi **`62.02`** (**`+44.0%`**, `188/192`), omp **`44.38`** (**`+40.0%`**, `194/222`), leg `57.88` (`17/20`) | **ACCEPTED:** Beat both neighboring controls on EVERY speed metric (`moe_dn 21.5 -> 16.6ms`, `total round=93.5ms`), saved **`-6,146 MiB` (`-6.0 GiB`) net DRM** (`86,024MiB`), passed **`50% (44.8K)` & `90% (78.2K)` needle gates `3/3`** (`~1,102 tok/s` prefill). Trade-off: `leg` dropped `20/20 -> 17/20` (soft fails) and prefill `~1102` vs `~1636` tok/s. |
| **Window 14 Winner: `v14-w32moe` (`0df9aa7`)** | `v13-iq4all` + true `wf=32` `FAST_MOE_GU=1` (`moe_wave32.hip.cpp` without `-mwavefrontsize64`) ([verdict-v14.txt]) | `79.9` (`+29.3%`) | `100.5` / `102.9`<br>*(q8fast: **`102.6` / `104.3`**)* | `39/40` (`hf=0`, Pi `67.0`, omp `47.6`, `gm=71.1`) | **`397/434`** (`hf=0`, `hr7=pass`, `625s`), Pi **`62.00`** (`91.3ms/r`, `188/192`), omp **`44.37`** (`84.6ms/r`, `191/222`), leg `18/20` | **ACCEPTED (lossless kernel win):** Fixed `-mwavefrontsize64` bug (`moe_gu +8.2%` @ `T=16`, `max_err=0`), dropped `fwd` to **`78.5ms`** (`round 92.9ms`), `Pi ms/r` `93.4 -> 91.3ms`, `50%/90%` cold needles **`3/3` @ `1,105 tok/s`**. `v14-q8fast` gave `hc_up 1.577×` & `shexp_dn 1.441×` (record `C3s=104.3`) but regressed `lm_head` `0.895×` (`3.2->3.6ms`); gating `FAST_Q8_WMMA` to `chunk<=24` keeps the win without the `lm_head` hit. `v14-q4kfast` rejected (`0.817×`). |
| **Window 15 Winner: `v15-ml-w15` (`73a1fec`)** | **Stock-MoE (MoE-lossless)** `v12-p2-p1` + `FAST_MOE_GU=1` (`wf=32`) + `FAST_Q8_WMMA=1` (`chunk<=24`) ([verdict-v15.txt]) | **`77.1`** (`+24.8%`, **`93–96%` of 4K ceil**) | **`94.5` / `100.5`** (`+30.5%` / `+35.3%`) | `37/40` (`hf=0`, Pi `61.6`, omp `44.3`, `gm5=72.5`) | **`408/434` (`94.0%`, `hf=0`, `hr7=pass`, `603s`)**, Pi **`59.86`** (**`+39.0%`**, `188/192`), omp **`43.06`** (**`+35.8%`**, **`200/222`**), leg **`20/20` (`100%`)** | **PRODUCTION CANDIDATE:** All-time best 434 accuracy (**beats Stock `403/434 hf=1` by +5 passes with 0 hard fails**); `lm_head` stayed `3.2ms` while `hc 5.8->5.4ms` & `shexp 2.2->2.0ms`; cold needles **`3/3` @ `1,615 tok/s` (`44.9K`) & `1,598 tok/s` (`78.2K`)**. Pure 100% weight-lossless (`v15-ll-w15`) also beat `v15-ll-ctrl` across every metric (`gm5 66.2 vs 64.1`). |

---

## Next Step (Window 15 / Production Promotion)

1. **Optional 1-Line Polish (`FAST_Q8_WMMA` gated to `chunk <= 24` + stacked with `FAST_MOE_GU=1`):**
   - Keeps `hc_up 1.577×` (`-0.4 ms/fwd`), `shexp_dn 1.441×` (`-0.2 ms/fwd`), `router` (`-0.1 ms/fwd`), and true `wf=32` `moe_gu` (`-1.0 ms/fwd`) while leaving `lm_head` on its `3.2 ms` (`98.3%` copy BW) kernel.
2. **Production Promotion Qualification (`references/publishing.md`):**
   - We are now at **90–92% of C1 hardware ceiling** (`80.2 / 87–89 tok/s`), **91.3% of copy BW on `moe_gu`**, **98.3% on `lm_head`**, and **100%+ on `q4k_draft` (`235.6 GB/s`)**.
   - Clean-checkout pin build + C=3 87K-context soak + user choice between **MoE-lossless** (`v12-p2-p1` + `w32moe` + `q8fast(<=24)`) vs **max-speed `IQ4_NL`** (`v14-w32moe` + `q8fast(<=24)`).
