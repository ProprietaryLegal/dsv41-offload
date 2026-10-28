"""Coordinator runs this on EACH Spark in a window; stdlib only, bounded duration."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


# Supported names are probed read-only before a window (nvidia-smi --help-query-gpu); never silently omitted.
FIELDS = ('uuid,clocks.current.graphics,clocks.current.memory,power.draw,power.limit,'
          'temperature.gpu,utilization.gpu,clocks_event_reasons.active')


# This file is authored/tested on CPU; its main is never run by the PF1-R2 lane.
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=int, default=7200)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.seconds <= 10800:
        ap.error('choose 1..10800 seconds')
    fields = FIELDS
    stop = time.monotonic() + args.seconds
    with args.output.open('x') as stream:
        while time.monotonic() < stop:
            started = time.monotonic()
            mem = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                key, _, value = line.partition(':')
                if key in ('MemAvailable', 'SwapFree', 'SwapTotal'):
                    mem[key] = int(value.split()[0])
            result = subprocess.run(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=5, check=True)
            row = {'utc_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
                   'memory_kib': mem, 'gpu_columns': fields, 'gpu_csv': result.stdout.strip()}
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            if mem['MemAvailable'] < 1572864:
                raise RuntimeError('MemAvailable below 1.5 GiB: coordinator stop rule; restore production')
            time.sleep(max(0, 1 - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
