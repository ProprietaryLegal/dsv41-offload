#!/usr/bin/env bash
# ./logs.sh [head|worker] [lines]   (add -f as 3rd arg to follow)
cd "$(dirname "$0")"; set -a; . ./dsv41.env; set +a
n=$([ "${1:-head}" = worker ] && echo $WORKER || echo $HEAD)
ssh $n "docker logs ${3:-} --tail ${2:-80} dsv41-serve 2>&1"
