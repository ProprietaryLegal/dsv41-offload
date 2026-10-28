#!/usr/bin/env bash
# Reviewed K1 bytes and R1 switch set; capacity is an ESTIMATED allocator result
# (offload/kvpack/kv_arith.py), to be confirmed by the new engine's own log.
K1_MANIFEST_SHA=559c5867ba42aa7daa7b4a7c8231d15ee1d35be40ae5b715b013f973193c58de
K1_EXPECT='VLLM_DSV41_KV_PAIR=3 VLLM_DSV41_DROP_MTP_BUFFER=1 VLLM_EXL3_TF_PROMPT_TRANSIENT=1 VLLM_EXL3_TF_PROMPT_ALIAS=1 VLLM_INDEXER_PREFILL_BUFFER_FACTOR=4'
K1_KV_BYTES=6442450944

# Call explicitly after sourcing the applied dsv41.env; staging alone needs no cap.
k1_select_profile() {
  case "${1:-}" in
    327680) K1_MAX_LEN=327680; K1_KV_TOKENS=2502600 ;;
    524288) K1_MAX_LEN=524288; K1_KV_TOKENS=2820756 ;;
    *) echo "K1 requires an explicit profile: 327680 or 524288" >&2; return 1 ;;
  esac
}
