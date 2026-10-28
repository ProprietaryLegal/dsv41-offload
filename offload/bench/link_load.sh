#!/usr/bin/env bash
# Run on the controller checkout (Controller); no installations on remote hosts.
set -euo pipefail
source "$HOME/.config/ndm/paths.sh"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/s1_load.py" "$@"
