"""Resume S2's interrupted shard 48 during idle windows, preserving shard 47."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
from pathlib import Path

from copy_to_x86host import ROOT, CopyInterrupted, copy, idle


def main() -> None:
    source = json.loads(Path(sys.argv[1]).read_text())
    previous = []
    shard47 = "model-00047-of-00048.safetensors"
    for line in Path(sys.argv[2]).read_text().splitlines():
        if line.startswith('{'):
            row = json.loads(line)
            if Path(row["file"]).name == shard47:
                previous.append(row)
    if len(previous) != 1:
        raise RuntimeError("expected one previously verified shard 47 receipt")
    entry47 = source["manifest"][shard47]
    if previous[0]["sha256"] != entry47["sha256"] or (ROOT/"engram"/shard47).stat().st_size != entry47["size"]:
        raise RuntimeError("previous shard 47 provenance/size differs")
    for filename in ("model-00048-of-00048.safetensors", "model.safetensors.index.json"):
        expected = source["manifest"][filename]["sha256"] if filename.endswith(".safetensors") else source["metadata_sha256"][filename]
        size = source["manifest"][filename]["size"] if filename.endswith(".safetensors") else None
        waiting = False
        while True:
            try:
                idle()
                row = copy(source["source_root"]+"/"+filename, ROOT/"engram"/filename, expected, size, resume=True)
                previous.append(row)
                break
            except (CopyInterrupted, urllib.error.URLError, TimeoutError) as error:
                if not waiting or isinstance(error, CopyInterrupted) and "copy aborted" in str(error):
                    print(f"WAIT_IDLE: {error}", flush=True)
                waiting = True
                time.sleep(0.25)
    (ROOT/"engram"/"ENGRAM_COPY_DONE").write_text(json.dumps({"files": previous}, indent=2)+"\n")
    print("ENGRAM_COPY_DONE", flush=True)


if __name__ == "__main__":
    main()
