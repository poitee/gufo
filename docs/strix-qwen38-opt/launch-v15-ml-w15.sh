#!/usr/bin/env bash
# Exact benchmarked launch script for Qwen3.8-Flash-Next on AMD Strix Halo (gfx1151).
# Default profile: v15-ml-w15 (MoE-lossless, 408/434 tool-call pass, 0 hard fails, C1 77.1, C3s 100.5, Pi 59.86).
# Set PURE_STOCK_WEIGHTS=1 for v15-ll-w15 (100% untouched stock target weights, no Q4K_DENSE).
set -euo pipefail

GUFO_BIN="${GUFO_BIN:-gufo}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the directory containing Qwen3.8-Flash-Next GGUF files}"
CACHE_DIR="${CACHE_DIR:-/tmp/gufo-qwen38-cache}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8411}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen38-gufo}"
PURE_STOCK_WEIGHTS="${PURE_STOCK_WEIGHTS:-0}"

mkdir -p "$CACHE_DIR"

# Core lossless kernel, graph, grammar, and MTP flags (verified 6/6 greedy C1 SHA parity)
export GUFO_STRIX_WIDE_BATCH=1
export GUFO_STRIX_Q4K_DRAFT=1
export GUFO_STRIX_Q4K_MTP=1
export GUFO_STRIX_MTP_GATE=1
export GUFO_STRIX_MTP_SAMP_DROP=0.55
export GUFO_STRIX_MTP_MAX_C2=0
export GUFO_STRIX_MTP_MAX_C3=5
export GUFO_STRIX_SPLITK_Q8=1
export GUFO_STRIX_BATCH_GRAPH=1
export GUFO_STRIX_GPU_GRAMMAR_MASK=1
export GUFO_STRIX_FAST_MOE_GU=1
export GUFO_STRIX_FAST_Q8_WMMA=1

if [ "$PURE_STOCK_WEIGHTS" != "1" ]; then
  # Opt-in Q4_K copies for small-batch dense projections (~3% of params; MoE experts stay stock Q5_1/Q8_0)
  export GUFO_STRIX_Q4K_SELFTEST=1
  export GUFO_STRIX_Q4K_DENSE="ssm_in,ssm_out,attn_qkv,attn_out,hc_down,shexp"
else
  unset GUFO_STRIX_Q4K_DENSE GUFO_STRIX_Q4K_SELFTEST || true
fi

exec "$GUFO_BIN" serve --host "$HOST" --port "$PORT" llm \
  --model "$MODEL_DIR/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --sessions 3 \
  --context 87040 \
  --reasoning-budget 2048 \
  --cache-ram-bytes 12884901888 \
  --cache-disk "$CACHE_DIR" \
  --cache-disk-bytes 34359738368 \
  --stall-timeout-ms 120000 \
  --speculative mtp \
  --mtp-model "$MODEL_DIR/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
