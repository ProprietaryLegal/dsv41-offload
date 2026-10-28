#!/usr/bin/env bash
# Run ON x86host after building ofl_loopback here. Own listener and synthetic CPU data only.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
cd "$HOME/dsv41-offload/r1"
ip -o -4 addr show | awk '$4 == "10.202.28.1/30" {found=1} END {exit !found}'
./ofl_loopback 18671 300 0 2000000 -1 -1 > loopback-burst-unpinned.jsonl 2> loopback-burst-unpinned.stderr
./ofl_loopback 18672 300 0 2000000 2 3 > loopback-burst-pinned.jsonl 2> loopback-burst-pinned.stderr
./ofl_loopback 18673 60 42000 2000000 2 3 > loopback-paced-spin.jsonl 2> loopback-paced-spin.stderr
./ofl_loopback 18674 60 42000 1 2 3 > loopback-paced-event.jsonl 2> loopback-paced-event.stderr
printf 'All loopback cases passed\n' > loopback.done
