"""ACC-SPLIT sources: one observation per measured request group, from bench2 runs and window cells (read-only).

Two families of files carry live drafter acceptance on this campaign:

- bench2 jsonl (offload/bench/bench2.py, 2026-10-22..24): a `run` row with the server's spec-decode counter deltas
  (`spec`) and one `request` row per stream with `prompt_sha256`, token counts, SSE step gaps and per-kind first/last
  times. Copies of the same file sit in many checkouts, so runs are de-duplicated by `run_id`.
- window cells (2026-10-26..27): SP1-style `cells/<cell>.json` and NBW-style `<start>-<cell>.json` carry `spec` and
  per-stream `prompt_sha256`; PC1/UP3/DEP1-style `<start>-<cell>.json` carry `repetitions`, each with Prometheus
  `metrics_before`/`metrics_after` and per-stream `prompt_sha256`.

Accepted means draft tokens accepted per speculative round (bonus token excluded), from the server counters
`spec_decode_num_accepted_tokens / spec_decode_num_drafts`; depth is `num_draft_tokens / num_drafts`. Nothing here reads
or writes prompt or generated text: prompts are identified by their sha256 only.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from offload.prefill_maxlen.evidence import Json, record

DRAFTS = 'vllm:spec_decode_num_drafts_total'
DRAFT_TOKENS = 'vllm:spec_decode_num_draft_tokens_total'
ACCEPTED = 'vllm:spec_decode_num_accepted_tokens_total'
PER_POS = 'vllm:spec_decode_num_accepted_tokens_per_pos_total'
_POSITION = re.compile(r'position="([0-9]+)"')
_WINDOW_TS = re.compile(r'(20[0-9]{6})T([0-9]{6})Z')
SKIP_CELLS = ('greedy', 'identity', 'warmup')


@dataclass(frozen=True)
class Obs:
    """One request group (one stream at B1) with server counter deltas over its run."""
    source: str                     # 'bench2' | 'cell'
    where: str                      # run_id (bench2) or '<window>/<start>/<cell>#<repeat>'
    date: str                       # YYYY-MM-DD (UTC)
    label: str                      # bench2 config_label or window name
    role: str                       # window start role ('control', 'K', 'I', ...) or 'bench2'
    cell: str                       # bench2 '<workload>-<ctx>' or the window cell name
    batch: int
    cache: str                      # bench2 cache_state ('cold' | 'warm_repeat' | ...); '' for window cells
    prompts: tuple[str, ...]        # prompt sha256 per stream
    prompt_tokens: tuple[int, ...]
    max_tokens: int
    reasoning_tokens: int           # summed over streams
    answer_tokens: int
    rounds: float
    draft_tokens: float
    accepted: float
    per_position: tuple[float, ...]  # accepted at positions 0..4 (unconditional counts over the run)
    ms_per_round: float | None = None
    decode_tps: float | None = None          # mean over streams, client side
    step_gap_mean_ms: float | None = None    # bench2 SSE gap (B1)
    step_gap_p50_ms: float | None = None
    step_events: int | None = None
    segments: Mapping[str, tuple[int, int]] = field(default_factory=dict)  # kind -> (tokens, SSE events), B1 only
    segment_tps: Mapping[str, float] = field(default_factory=dict)        # kind -> client decode tok/s, B1 only

    @property
    def accepted_per_round(self) -> float:
        return self.accepted / self.rounds

    @property
    def depth(self) -> float:
        return self.draft_tokens / self.rounds

    @property
    def prompt(self) -> str:
        """One identity for the request group: the stream's sha256 at B1, a hash of the sorted shas otherwise."""
        if len(self.prompts) == 1:
            return self.prompts[0]
        return 'group:' + hashlib.sha256('|'.join(sorted(self.prompts)).encode()).hexdigest()


def _num(value: Json, default: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return default
    return float(value)


def _int(value: Json) -> int:
    v = _num(value)
    return int(v) if v is not None else 0


def _per_position(spec: Mapping[str, Json]) -> tuple[float, ...]:
    pos = spec.get('accepted_per_position')
    if not isinstance(pos, dict):
        return ()
    values = {int(k): _num(v, 0.0) or 0.0 for k, v in pos.items()}
    return tuple(values.get(i, 0.0) for i in range(max(values) + 1)) if values else ()


def _counter(metrics: Mapping[str, Json], name: str) -> float:
    return sum(_num(v, 0.0) or 0.0 for k, v in metrics.items() if k == name or k.startswith(name + '{'))


def _positions(metrics: Mapping[str, Json]) -> dict[int, float]:
    out: dict[int, float] = {}
    for k, v in metrics.items():
        if k.startswith(PER_POS + '{'):
            m = _POSITION.search(k)
            if m:
                out[int(m.group(1))] = out.get(int(m.group(1)), 0.0) + (_num(v, 0.0) or 0.0)
    return out


def _segments(req: Mapping[str, Json]) -> dict[str, tuple[int, int]]:
    """(tokens, SSE events) per output kind from the per-kind first/last times and the SSE step times (B1, bench2)."""
    times = req.get('step_times_ms')
    first, last = req.get('first_by_kind_s'), req.get('last_by_kind_s')
    if not isinstance(times, list) or not isinstance(first, dict) or not isinstance(last, dict):
        return {}
    stamps = [t / 1000.0 for t in (_num(x) for x in times) if t is not None]
    tokens = {'reasoning': _int(req.get('reasoning_tokens')), 'answer': _int(req.get('answer_tokens'))}
    out: dict[str, tuple[int, int]] = {}
    for kind, n in tokens.items():
        lo, hi = _num(first.get(kind)), _num(last.get(kind))
        if n <= 0 or lo is None or hi is None:
            continue
        events = sum(1 for t in stamps if lo - 1e-6 <= t <= hi + 1e-6)
        if events > 0:
            out[kind] = (n, events)
    return out


def bench2_files(roots: Iterable[Path]) -> list[Path]:
    """bench2 jsonl files under the roots, one per distinct (size, first line): checkouts hold many copies."""
    seen: dict[tuple[int, str], Path] = {}
    for root in roots:
        for path in sorted(root.rglob('*.jsonl')):
            try:
                with path.open() as f:
                    first = f.readline()
                row = json.loads(first)
            except (OSError, ValueError):
                continue
            if not isinstance(row, dict) or 'run_id' not in row or 'config_label' not in row:
                continue
            seen.setdefault((path.stat().st_size, hashlib.sha256(first.encode()).hexdigest()), path)
    return sorted(seen.values())


def load_bench2(paths: Iterable[Path]) -> list[Obs]:
    """One Obs per bench2 run (de-duplicated by run_id) that has server counters and request rows."""
    runs: dict[str, dict[str, Json]] = {}
    requests: dict[str, list[dict[str, Json]]] = {}
    for path in paths:
        with path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict) or not isinstance(row.get('run_id'), str):
                    continue
                rid = cast(str, row['run_id'])
                if row.get('type') == 'run':
                    runs.setdefault(rid, cast(dict[str, Json], row))
                elif row.get('type') == 'request':
                    requests.setdefault(rid, [])
                    sha = row.get('prompt_sha256')
                    if not any(r.get('prompt_sha256') == sha and r.get('request_index') == row.get('request_index')
                               for r in requests[rid]):
                        requests[rid].append(cast(dict[str, Json], row))
    out: list[Obs] = []
    for rid, run in sorted(runs.items()):
        spec, reqs = run.get('spec'), requests.get(rid, [])
        if not isinstance(spec, dict) or not reqs or any(r.get('error') for r in reqs):
            continue
        rounds = _num(spec.get('rounds'), 0.0) or 0.0
        if rounds <= 0:
            continue
        settings = record(reqs[0].get('settings') or {})
        gap = reqs[0].get('step_gap_ms') if len(reqs) == 1 else None
        gaps = record(gap) if isinstance(gap, dict) else {}
        tps = [t for t in (_num(r.get('decode_tps')) for r in reqs) if t is not None]
        out.append(Obs(
            source='bench2', where=rid, date=str(run.get('ts', ''))[:10], label=str(run.get('config_label')),
            role='bench2', cell=f"{run.get('workload')}-{run.get('target_ctx')}", batch=len(reqs),
            cache=str(run.get('cache_state') or ''),
            prompts=tuple(str(r.get('prompt_sha256')) for r in reqs),
            prompt_tokens=tuple(_int(r.get('prompt_tokens')) for r in reqs),
            max_tokens=_int(settings.get('max_tokens')),
            reasoning_tokens=sum(_int(r.get('reasoning_tokens')) for r in reqs),
            answer_tokens=sum(_int(r.get('answer_tokens')) for r in reqs),
            rounds=rounds, draft_tokens=_num(spec.get('proposed_tokens'), 0.0) or 0.0,
            accepted=_num(spec.get('accepted_tokens'), 0.0) or 0.0, per_position=_per_position(spec),
            decode_tps=sum(tps) / len(tps) if tps else None,
            step_gap_mean_ms=_num(gaps.get('mean')), step_gap_p50_ms=_num(gaps.get('p50')),
            step_events=_int(reqs[0].get('step_events')) if len(reqs) == 1 else None,
            segments=_segments(reqs[0]) if len(reqs) == 1 else {},
            segment_tps={k: v for k, v in (('reasoning', _num(reqs[0].get('reasoning_decode_tps'))),
                                           ('answer', _num(reqs[0].get('answer_decode_tps')))) if v} if len(reqs) == 1 else {}))
    return out


def _window_date(name: str) -> str:
    m = _WINDOW_TS.search(name)
    return f'{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}' if m else ''


def _streams(body: Mapping[str, Json]) -> list[dict[str, Json]]:
    streams = body.get('streams')
    return [record(s) for s in streams if isinstance(s, dict)] if isinstance(streams, list) else []


def _stream_tokens(s: Mapping[str, Json]) -> tuple[int, int, int, int]:
    """(prompt, max_tokens, reasoning, answer) of one window stream; both window layouts."""
    usage = record(s.get('usage') or {}) if isinstance(s.get('usage'), dict) else {}
    details = usage.get('completion_tokens_details')
    reasoning = _int(s.get('reasoning_tokens')) or _int(record(details).get('reasoning_tokens') if isinstance(details, dict) else None)
    answer = _int(s.get('answer_and_control_tokens'))
    sampling = s.get('sampling')
    budget = _int(record(sampling).get('max_tokens')) if isinstance(sampling, dict) else 0
    prompt = _int(s.get('prompt_tokens')) or _int(usage.get('prompt_tokens'))
    return prompt, budget, reasoning, answer


def _cell_obs(window: str, start: str, role: str, cell: str, repeat: int, body: Mapping[str, Json],
              default_max: int) -> Obs | None:
    streams = _streams(body)
    if not streams or any(not isinstance(s.get('prompt_sha256'), str) for s in streams):
        return None
    tokens = [_stream_tokens(s) for s in streams]
    spec = body.get('spec')
    mb, ma = body.get('metrics_before'), body.get('metrics_after')
    if isinstance(spec, dict) and (_num(spec.get('rounds'), 0.0) or 0.0) > 0:
        rounds = _num(spec.get('rounds'), 0.0) or 0.0
        drafts = _num(spec.get('proposed_tokens'), 0.0) or 0.0
        accepted = _num(spec.get('accepted_tokens'), 0.0) or 0.0
        per_pos = _per_position(spec)
    elif isinstance(mb, dict) and isinstance(ma, dict):
        rounds = _counter(ma, DRAFTS) - _counter(mb, DRAFTS)
        drafts = _counter(ma, DRAFT_TOKENS) - _counter(mb, DRAFT_TOKENS)
        accepted = _counter(ma, ACCEPTED) - _counter(mb, ACCEPTED)
        before, after = _positions(mb), _positions(ma)
        per_pos = tuple(after.get(i, 0.0) - before.get(i, 0.0) for i in range(max(after, default=-1) + 1))
    else:
        return None
    if rounds <= 0:
        return None
    ms = _num(body.get('server_decode_ms_per_spec_request_round'))
    if ms is None and _num(body.get('server_decode_ms')) and (_num(body.get('spec_request_rounds'), 0.0) or 0.0) > 0:
        ms = (_num(body.get('server_decode_ms'), 0.0) or 0.0) / (_num(body.get('spec_request_rounds'), 1.0) or 1.0)
    tps = [t for t in (_num(s.get('decode_tps')) for s in streams) if t is not None]
    budgets = {t[1] for t in tokens if t[1] > 0}
    return Obs(
        source='cell', where=f'{window}/{start}/{cell}#{repeat}', date=_window_date(window), label=window,
        role=role, cell=cell, batch=len(streams), cache='',
        prompts=tuple(str(s['prompt_sha256']) for s in streams), prompt_tokens=tuple(t[0] for t in tokens),
        max_tokens=budgets.pop() if len(budgets) == 1 else default_max,
        reasoning_tokens=sum(t[2] for t in tokens), answer_tokens=sum(t[3] for t in tokens),
        rounds=rounds, draft_tokens=drafts, accepted=accepted, per_position=per_pos, ms_per_round=ms,
        decode_tps=sum(tps) / len(tps) if tps else _num(body.get('median_decode_tps')))


def _roles(summary: Mapping[str, Json]) -> dict[str, str]:
    starts = summary.get('starts')
    if not isinstance(starts, dict):
        return {}
    return {k: str(record(v).get('role')) if isinstance(v, dict) else '' for k, v in starts.items()}


def _cells(window: Path) -> Iterator[tuple[str, str, str, Path]]:
    """(start, role, cell, path) for every cell file of a window, both layouts."""
    try:
        summary = record(cast(Json, json.loads((window / 'summary.json').read_text())))
    except (OSError, ValueError):
        summary = {}
    roles = _roles(summary)
    for start, role in roles.items():
        for path in sorted(window.glob(f'{start}-*.json')):
            cell = path.name[len(start) + 1:-len('.json')]
            if cell not in SKIP_CELLS:
                yield start, role, cell, path
    cells_dir = window / 'cells'
    if cells_dir.is_dir():                      # SP1: one implicit control start
        for path in sorted(cells_dir.glob('*.json')):
            if path.stem not in SKIP_CELLS:
                yield 'sp1', 'control' if path.stem.startswith('ctl-') else 'profiled', path.stem, path


def window_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and _WINDOW_TS.search(p.name)
                  and ((p / 'summary.json').exists() or (p / 'cells').is_dir()))


def load_windows(dirs: Iterable[Path], default_max: int = 0) -> list[Obs]:
    out: list[Obs] = []
    for window in dirs:
        for start, role, cell, path in _cells(window):
            try:
                body = record(cast(Json, json.loads(path.read_text())))
            except (OSError, ValueError):
                continue
            reps = body.get('repetitions')
            rows = [record(r) for r in reps if isinstance(r, dict)] if isinstance(reps, list) and reps else [body]
            for i, rep in enumerate(rows):
                obs = _cell_obs(window.name, start, role, cell, i, rep, default_max)
                if obs is not None:
                    out.append(obs)
    return out
