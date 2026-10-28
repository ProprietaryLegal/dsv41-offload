"""Inventory pinned donor history as data; never import or execute donor code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import TypedDict

HERE = Path(__file__).resolve().parent
DONOR = Path('/opt/dsv41/data/upstream/henryous')
START = 'b172664'
END = 'afe4d5e41ec6fa4db17c5cc88758af439a106913'

# Each decision covers the patch AND its corresponding installed-source mirror.
DECISIONS: dict[str, tuple[str, str]] = {
    '0040': ('UP1 config candidate', 'K5 at 3/4 streams fits target mrow32 and remote block5; must remeasure remote full-round cost, not import local timings.'),
    '0051': ('K1', 'Target indexer defer; use K1 composition, no second mount.'),
    '0053': ('K1/off', 'Display KV code carried by K1; switch stays off, slower reads and less headroom.'),
    '0054': ('reject', 'Display embedding/Markov allocation regressed; Markov is remote and embedding move conflicts with capacity accounting.'),
    '0055': ('reject', 'Hybrid local drafter probability/state path is absent in capacity mode; remote contract is greedy. Also failed upstream prose gate.'),
    '0056': ('RP1', 'CRC protects files, not rank mismatch; retain RP1 safety gates.'),
    '0057': ('K1', 'FP8 page pairing; preserve K1 record/group layout.'),
    '0058': ('RP1', 'Connector-specific allocator exception, not a general allocator change.'),
    '0059': ('K1/off', 'NVFP4 code dependency carried by K1; NEVER enable, distribution regression.'),
    '0060': ('K1', 'Three-page FP8 packing; scheduler/cache boundaries change, same target records.'),
    '0061': ('reject', 'Compressed NVFP4 remains above KLD floor and loses accepted tokens; do not alter target quantization.'),
    '0062': ('reject', 'Unmeasured 0.03–0.25% sampling-only estimate; host mask/event sync and async process parked stream need separate design. No positive result justifies port.'),
    '0063': ('reject', 'Measured regression at target verify rows 17–32; competes with routed expert reads. Remote drafter does not fix target bandwidth.'),
    '0064': ('UP1 overlay', 'Target MoE 33–63 row split applies to mixed steps regardless of remote drafter. Changes rounding; affinity must be isolated.'),
    '0065': ('reject', 'No sustained improvement; 446 MiB extra graphs, breaks memory budget and async graph assumptions.'),
    '0066': ('reject local path', 'Local DSpark fused/trim/prefetch never executes in remote capacity mode; x86host already has its own fused Markov. No SM121 code port to SM86.'),
    '0067': ('UP1 overlay', 'Target SWA metadata and replay upload remain local; compose model_state with X1 0075, keep no local-drafter changes.'),
    '0071': ('SO1', 'Remote depth cap needs remote policy; SO1 selected scheduler pause instead, so do not reintroduce local speculator.'),
    '0072': ('SO1', 'Priority-aware background pause adapted to our priority5 floor.'),
    '0073': ('SO1', 'Waiting foreground throttle in SO1 scheduler; no duplicate scheduler mount.'),
    '0074': ('SO1', 'Solo background chunks in SO1 scheduler; compose 0075 hooks before combined test.'),
    '0075': ('UP1 via X1', 'Use provenance-aware X1 port, never raw donor: remote partial-prefill taps must carry actual suffix length. Scheduler and worker sides inseparable.'),
    '0076': ('K1', 'Memory switches/memprobe already carried by K1; preserve all K1 runner bytes.'),
    '0076c': ('reject', 'Unused deterministic-prefill tile sidepath; TF_PROMPT=1 bypasses it at >=64 rows, <=63 handled by 0064; association changes without a positive measurement.'),
    '0077': ('RP1', 'Two-node store/load requires identity/layout and byte-equality gates; independent of remote draft wire state.'),
    '0078': ('DROP this round', 'S17: ungated C++ changes, hint-timeout escalation and row-split overhead. X1 code retained unmounted for reference only; no arm or new Engram JIT.'),
    '0079': ('reject', 'Target mechanisms apply but measured within noise; cache lifetime/aliasing and K1 flashinfer composition risk without gain.'),
    '0080': ('reject', 'Dummy full-context startup reservation reduced lowest free memory by 280 MiB; remote parked streams add warmup hazards.'),
    '0081': ('RP1', 'Load cap/I/O affinity; RP1 handles connector and headroom tests.'),
}


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(DONOR), *args], text=True)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Change(TypedDict):
    path: str
    status: str
    old_sha256: str | None
    new_sha256: str | None
    owners: list[str]
    classification: str


def generate() -> tuple[str, str]:
    if git('rev-parse', 'HEAD').strip() != END or git('status', '--porcelain').strip():
        raise ValueError('donor must be clean at pinned afe4d5e')
    readme = git('show', f'{END}:README.md')
    patches = sorted(p for p in (DONOR/'patches').glob('*.patch') if p.stem.split('-')[0] in DECISIONS)
    file_owners: dict[str, list[str]] = {}
    rows: list[dict[str, object]] = []
    for patch in patches:
        key = patch.stem.split('-')[0]
        content = patch.read_text()
        for name in re.findall(r'^\+\+\+ b/(\S+)', content, re.M):
            file_owners.setdefault('docker/files/'+name, []).append(key)
        quotes = [line for line in readme.splitlines() if line.startswith('| '+key+' |')]
        if key == '0076c':
            quotes = [line for line in readme.splitlines() if '0062, 0063, 0065 and 0076c' in line]
            if not quotes:
                quotes = ['No measured result or adoption claim in README; Dockerfile explicitly excludes 0076c.']
        if len(quotes) != 1:
            raise ValueError(f'missing/ambiguous README row: {key}')
        owner, decision = DECISIONS[key]
        rows.append({'id': key, 'patch': str(patch.relative_to(DONOR)), 'sha256': digest(patch.read_bytes()),
                     'disposition': owner, 'our_applicability_and_conflicts': decision,
                     'upstream_readme_verbatim': quotes[0]})
    changes: list[Change] = []
    for line in git('diff', '--name-status', START, END).splitlines():
        status, name = line.split('\t')
        owners = file_owners.get(name, [])
        if name.startswith('patches/'):
            key = name.split('/')[1].split('-')[0]
            if key not in DECISIONS:
                raise ValueError(f'uncatalogued patch: {name}')
            owners = [key]
            classification = 'patch source/config documentation'
        elif name.startswith('docker/files/'):
            if not owners:
                raise ValueError(f'uncatalogued installed-source change: {name}')
            classification = 'installed-source mirror; includes recipe baseline for newly tracked files'
        elif name == 'README.md':
            classification = 'documentation: measurements, limitations, topology incident, licensing/credits; never a local measurement'
        elif name == 'docker/Dockerfile':
            classification = 'v24 COPY image composition; do not upgrade image/base or execute donor builds'
        elif name == 'serve/serve_example.sh':
            classification = 'configuration: see CATALOGUE.md non-patch table; never run donor launcher'
        else:
            raise ValueError(f'unclassified non-patch change: {name}')
        old = None if status == 'A' else digest(subprocess.check_output(['git','-C',str(DONOR),'show',f'{START}:{name}']))
        new = None if status == 'D' else digest(subprocess.check_output(['git','-C',str(DONOR),'show',f'{END}:{name}']))
        changes.append({'path': name, 'status': status, 'old_sha256': old, 'new_sha256': new,
                        'owners': owners, 'classification': classification})
    commits = [{'sha': line.split(' ',1)[0], 'subject': line.split(' ',1)[1],
                'paths': git('diff-tree','--no-commit-id','--name-only','-r',line.split(' ',1)[0]).splitlines()}
               for line in git('log','--reverse','--format=%H %s',f'{START}..{END}').splitlines()]
    payload = {'range': [git('rev-parse',START).strip(), END], 'readme_sha256': digest(readme.encode()),
               'commits': commits, 'patches': rows, 'changed_paths': changes}
    rendered = '# Complete upstream patch catalogue\n\nGenerated from the pinned local reference. Verbatim README claims belong to upstream, not our stack.\n\n'
    rendered += '| Patch | Disposition | Our applicability / conflict | Upstream README (verbatim) |\n|---|---|---|---|\n'
    for row in rows:
        quote = str(row['upstream_readme_verbatim']).replace('|',' / ')
        rendered += f"| {row['id']} | {row['disposition']} | {row['our_applicability_and_conflicts']} | {quote} |\n"
    rendered += '\n0040 is a changed config note, covered separately in CATALOGUE.md. No patches 0052/0068/0069/0070 exist in this range.\n'
    return json.dumps(payload, indent=2)+'\n', rendered



def audit_image(image: Path = Path('/opt/dsv41/data/src-image')) -> dict[str, object]:
    """Prove no additional code delta is hidden in the installed-source mirrors."""
    generate()  # donor identity, cleanliness and exhaustive path/README check first
    changed = [name for name in git('diff','--name-only',START,END).splitlines()
               if name.startswith('docker/files/')]
    patches = sorted(p for p in (DONOR/'patches').glob('*.patch')
                     if p.name[:4].isdigit() and 51 <= int(p.name[:4]) <= 81
                     and p.name[:4] not in ('0062','0063','0065') and not p.name.startswith('0076c'))
    matches: list[dict[str, str | None]] = []
    with tempfile.TemporaryDirectory(prefix='up1-image-audit-') as tmp:
        tree = Path(tmp)
        for name in changed:
            rel = name.removeprefix('docker/files/')
            source = image/rel
            if source.is_file():
                target = tree/rel
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(source.read_bytes())
        for patch in patches:
            proc = subprocess.run(['patch','--batch','--forward','--fuzz=0','--no-backup-if-mismatch',
                                   '-p1','-d',tmp,'-i',str(patch)],text=True,capture_output=True)
            if proc.returncode or 'fuzz' in proc.stdout.lower():
                raise ValueError(f'image audit patch failed: {patch.name}: {proc.stdout} {proc.stderr}')
        for name in changed:
            rel = name.removeprefix('docker/files/')
            source, built, upstream = image/rel, tree/rel, DONOR/name
            if built.read_bytes() != upstream.read_bytes():
                raise ValueError(f'unexplained installed-source delta: {rel}')
            matches.append({'path':rel,'image_base_sha256':digest(source.read_bytes()) if source.exists() else None,
                            'rebuilt_and_donor_sha256':digest(built.read_bytes())})
    return {'donor_sha':END,'fuzz':0,'patches':[p.name for p in patches],
            'identical_files':matches,'residual':[],
            'qualification':'Source reconstruction only; donor code was never executed'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--audit-image', action='store_true')
    args = parser.parse_args()
    for name, body in zip(('catalogue.json','PATCH-CATALOGUE.md'), generate(), strict=True):
        path = HERE/name
        if args.check:
            if path.read_text() != body:
                raise ValueError(f'{name} differs from pinned donor inventory')
        else:
            path.write_text(body)
    if args.audit_image:
        audit = json.dumps(audit_image(),indent=2)+'\n'
        path = HERE/'image-audit.json'
        if args.check:
            if path.read_text() != audit:
                raise ValueError('image reconstruction audit drift')
        else:
            path.write_text(audit)
    print('catalogue complete: seven commits, every changed path classified with before/after hashes')


if __name__ == '__main__':
    main()
