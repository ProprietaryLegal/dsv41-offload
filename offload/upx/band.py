"""R-UPX band cells and band probes: the 33-63-row MoE calls that 0064 changes (and that upstream measured CPU pin with).

Why the shared cells cannot decide 0064 (MEASURED usage, arithmetic from the v5 scheduler and the production SPEC):
- The six W5/K1 greedy prompts are 17-27 tokens (K1-production/greedy-post.jsonl usage) and a B1 decode step verifies at
  most 1 + 5 rows. No step of the standard six-prompt greedy gate reaches 33 rows, so that gate is blind to 0064.
- At four slots a decode step has at most 4 x (1 + 4) = 20 rows (SPEC [[1,2,5],[3,4,4]]). A long prompt beside other
  requests runs long_prefill_token_threshold = 1,024-token chunks (scheduler.py: the threshold applies with > 1 eligible
  request), a lone one 4,096-token chunks. A step lands in 33-63 rows only when a prefill piece of at most 63 rows minus
  the decode rows beside it runs: a short or prefix-cached prompt, or a prompt's final remainder. ctl-mixed150k-4user
  and ctl-32k-b4 therefore contain at most one such step per prompt (its remainder), by chance.

Band cells (BAND_CELLS) put one short token-id prompt of 36-48 tokens (max_tokens 1) into the fourth slot, one after
another, while three long decoders run. Each short prompt's prefill step has its own L rows plus 3-15 decode rows
(three decoders, k = 0..4), 39-63 rows whatever the decoders' k: one band step per short prompt, counted in the result
(`band_steps`), and the cell refuses below MIN_BAND_STEPS or MIN_BAND_FRACTION. The decision metric is the decoders'
pooled server ms per round (nbw_cells.summarize's definition), which includes every band step.

Band probes are NDM's 0.9-nat screen (greedy near-tie + registered KL, as UP2-ADOPT for 0075) aimed at the band:
token-id prompts whose ONE prefill step, or whose final chunk after full 4,096-row chunks, has 33/42/43/63 rows (the
0064 band edges: one exl3_smoe chunk up to 42 rows, two above), with 32 and 64 as unchanged-dispatch controls.
Nothing here contacts a host: a runner calls these functions inside its HOLD (WINDOWS.md).
"""
from __future__ import annotations

import hashlib
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, cast

from offload.gb10_budget import window_cells as wc
from offload.launch.windows import greedy_kl as gk
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, number, record
from offload.prefill_trace.window_requests import TEMPLATE

MODEL = 'DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2'
COOP_ROWS_MAX = 32          # exl3 coop decode path takes m <= 32 rows; 0064 leaves it alone
SPLIT_MAX = 63              # VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=63 (R-UP3 split arm); m >= 64 keeps TensorFold's prompt path
SMOE_CHUNK = 42             # smoe.max_rows() at top_k 6: 33-42-row calls are one exl3_smoe chunk, 43-63 two
BAND = (COOP_ROWS_MAX + 1, SPLIT_MAX)
PREFILL_CHUNK_B1 = 4096     # production max_num_batched_tokens: a lone prompt's chunks
DECODERS = 3
DECODE_ROWS = (DECODERS * 1, DECODERS * (1 + 4))   # beside a short prompt: 4 requests -> k <= 4; k = 0 without drafts
TINY_LENGTHS = (36, 40, 44, 48)
TINY_SPACING = 64           # token offset between short prompts in the tiny region
TINY_SLOTS = 256            # distinct short prompts; reused across repeats with fresh cache salts
MAX_TINY = 2000
MIN_BAND_STEPS = 30         # registration: a decision band cell needs at least this many band steps per repeat
MIN_BAND_FRACTION = .10     # ... and band steps >= 10 % of the decoders' rounds
PROBE_LENGTHS = (32, 33, 42, 43, 63, 64, 2 * PREFILL_CHUNK_B1 + 40, 8 * PREFILL_CHUNK_B1 + 40)
CONTROL_LENGTHS = (32, 64)  # unchanged dispatch under 0064: reported identity, same bound
GREEDY_LENGTHS = (33, 38, 42, 43, 52, 63)
LONG_GREEDY_LENGTHS = (32000, 150000)   # 0067 (VOR): decode at long context exercises every SWA group's metadata
GREEDY_MAX_TOKENS = 300
NEAR_TIE_NATS = 0.9 + 1e-3  # NDM 2026-10-27 ".9" (nbw_cells.greedy near_tie_nats registration)
KL_FLOOR = .005             # bound = max(0.005, 2 x the A/A floor): UP2's registration (up2_checks.numerical)
TTFT_TIMEOUT_S = 600.0
# Public corpus regions (characters into the body after HEADER), disjoint from window_cells.PROMPTS (0-4.5 M, 9 M):
# (offset, characters, tokens needed).
REGIONS = {'long': (5_000_000, 560_000, max(LONG_GREEDY_LENGTHS)),
           'probe': (6_000_000, 130_000, max(PROBE_LENGTHS)),
           'greedy': (6_400_000, 12_000, 100 * len(GREEDY_LENGTHS)),
           'tiny': (6_500_000, 70_000, TINY_SPACING * TINY_SLOTS)}


class Api(Protocol):
    def post(self, endpoint: str, body: dict[str, Json]) -> dict[str, Json]: ...
    def metrics(self) -> str: ...
    def model(self) -> dict[str, Json]: ...
    def stream(self, body: dict[str, Json], first: threading.Event) -> dict[str, Json]: ...


class Counters(Protocol):
    def parse_metrics(self, text: str) -> dict[str, float]: ...


def tiny_rows(length: int) -> tuple[int, int]:
    """Rows of the step that prefills a short prompt of `length` tokens beside DECODERS decoders."""
    return length + DECODE_ROWS[0], length + DECODE_ROWS[1]


def in_band(rows: int) -> bool:
    return BAND[0] <= rows <= BAND[1]


def probe_final_rows(length: int) -> int:
    """Rows of the last prefill step of a lone token-id prompt (B1: full 4,096-row chunks, then the remainder)."""
    return length % PREFILL_CHUNK_B1 or PREFILL_CHUNK_B1


# --- public token regions ------------------------------------------------------------------------------------------

def token_ids(value: Json) -> list[int]:
    ck.require(isinstance(value, list) and all(isinstance(t, int) and not isinstance(t, bool) for t in value),
               'token ids expected')
    return cast(list[int], value)


def regions(api: Api, materials: str) -> dict[str, list[int]]:
    text = wc.body(materials)
    out: dict[str, list[int]] = {}
    for name, (offset, chars, needed) in REGIONS.items():
        ck.require(offset + chars <= len(text), f'{name}: region outside the public corpus')
        ids = token_ids(api.post('/tokenize', {'model': MODEL, 'prompt': text[offset:offset + chars],
                                                'add_special_tokens': False})['tokens'])
        ck.require(len(ids) >= needed, f'{name}: {len(ids)} tokens, {needed} needed')
        out[name] = ids[:needed]
    return out


def ids_sha256(ids: Sequence[int]) -> str:
    return hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()


def region_pins(built: Mapping[str, Sequence[int]]) -> dict[str, Json]:
    return {name: {'tokens': len(ids), 'sha256': ids_sha256(ids)} for name, ids in sorted(built.items())}


def verify_regions(built: Mapping[str, Sequence[int]], pins: Mapping[str, Json]) -> None:
    """Every later start rebuilds the regions and must get the C1 start's token ids (tokenizer or corpus drift STOPs)."""
    ck.require(region_pins(built) == dict(pins), 'public token regions differ from the first start')


def tiny_prompt(tiny: Sequence[int], j: int) -> list[int]:
    offset = TINY_SPACING * (j % TINY_SLOTS)
    return list(tiny[offset:offset + TINY_LENGTHS[j % len(TINY_LENGTHS)]])


def greedy_prompts(region: Sequence[int]) -> list[list[int]]:
    return [list(region[100 * i:100 * i + n]) for i, n in enumerate(GREEDY_LENGTHS)]


def long_prompts(region: Sequence[int]) -> list[list[int]]:
    return [list(region[:n]) for n in LONG_GREEDY_LENGTHS]


# --- KL probes (A/A floor from the bracketing controls) -------------------------------------------------------------

def kl_probes(api: wc.Http, probe: Sequence[int], salt_prefix: str) -> dict[str, Json]:
    """Top-20 at every PROBE_LENGTHS prefix, fresh cache salt each: C1, X and C2 each run this once."""
    return {str(n): gk.probe(api, list(probe[:n]), f'{salt_prefix}:kl:{n}') for n in PROBE_LENGTHS}


def kl_screen(c1: Mapping[str, Json], candidate: Mapping[str, Json], c2: Mapping[str, Json]) -> dict[str, Json]:
    """KL(C1 || X) per length within max(0.005, 2 x KL(C1 || C2)), common mass >= 0.99 (greedy_kl.passes)."""
    rows: dict[str, Json] = {}
    for n in map(str, PROBE_LENGTHS):
        floor = gk.top20_kl(record(c1[n]), record(c2[n]))
        result = gk.top20_kl(record(c1[n]), record(candidate[n]))
        bound = max(KL_FLOOR, 2 * floor['kl_nats'])
        rows[n] = {'final_rows': probe_final_rows(int(n)), 'band': in_band(probe_final_rows(int(n))),
                   'aa_kl': floor['kl_nats'], 'kl_nats': result['kl_nats'], 'common_mass': result['common_mass'],
                   'aa_common_mass': floor['common_mass'], 'bound': bound,
                   'identical_top20': dict(record(c1[n])) == dict(record(candidate[n])),
                   'pass': gk.passes(result, bound) and floor['common_mass'] >= gk.MIN_COMMON_MASS}
    return {'rows': rows, 'pass': all(record(r)['pass'] is True for r in rows.values()),
            'registration': f'top20+OTHER KL(C1||X) <= max({KL_FLOOR}, 2 x KL(C1||C2)), common mass >= {gk.MIN_COMMON_MASS}'}


# --- token-id greedy (band prompts for 0064, long prompts for 0067) -------------------------------------------------

def generate(api: wc.Http, prompt: Sequence[int], salt: str, max_tokens: int = GREEDY_MAX_TOKENS) -> list[int]:
    r = api.post('/v1/completions', {'model': MODEL, 'prompt': cast(Json, list(prompt)), 'max_tokens': max_tokens,
                                     'temperature': 0, 'ignore_eos': True, 'logprobs': 1,
                                     'return_tokens_as_token_ids': True, 'priority': 10, 'cache_salt': salt})
    choice = record(ck.rows(r['choices'])[0])
    ck.require(choice.get('finish_reason') == 'length', 'greedy output did not run to its fixed length')
    tokens = record(choice['logprobs'])['tokens']
    ck.require(isinstance(tokens, list) and len(tokens) == max_tokens, 'greedy token list incomplete')
    out: list[int] = []
    for t in cast(list[Json], tokens):
        ck.require(isinstance(t, str) and t.startswith('token_id:') and t[9:].isdigit(), 'malformed output token id')
        out.append(int(cast(str, t)[9:]))
    return out


def near_tie(top: Mapping[str, Json], a: int, b: int, threshold: float = NEAR_TIE_NATS) -> dict[str, Json]:
    """nbw_cells.near_tie's rule on token ids: both tokens in the top-20, within `threshold` of each other and of the top."""
    lps = {int(str(k)[9:]): number(v) for k, v in top.items()}
    best = max(lps.values())
    ok = a in lps and b in lps and abs(lps[a] - lps[b]) <= threshold and max(lps[a], lps[b]) >= best - threshold
    return {'near_tie': ok,
            'margin_nats': None if a not in lps or b not in lps else round(abs(lps[a] - lps[b]), 4),
            'gap_to_top_nats': None if a not in lps or b not in lps else round(best - max(lps[a], lps[b]), 4)}


def token_greedy(api: wc.Http, prompts: Sequence[Sequence[int]], reference: Sequence[Sequence[int]] | None,
                 salt_prefix: str, *, threshold: float = NEAR_TIE_NATS) -> dict[str, Json]:
    """T=0 continuation of each prompt. reference None (the C1 start) records the outputs; otherwise every output must
    equal the reference or diverge first at a registered near-tie; the top-20 and prefix at each divergence are kept
    for greedy_kl.compare_recorded in the following control start."""
    ck.require(reference is None or len(reference) == len(prompts), 'one reference output per prompt')
    rows: list[Json] = []
    for i, prompt in enumerate(prompts):
        out = generate(api, prompt, f'{salt_prefix}:greedy:{i}')
        row: dict[str, Json] = {'i': i, 'prompt_tokens': len(prompt), 'sha256': ids_sha256(out),
                                'output_token_ids': cast(Json, out)}
        if reference is not None:
            ref = list(reference[i])
            same = out == ref
            row.update(identical=same, reference_sha256=ids_sha256(ref))
            if not same:
                j = next(k for k, (u, v) in enumerate(zip(ref, out)) if u != v)
                prefix = [*prompt, *ref[:j]]
                top = gk.probe(api, prefix, f'{salt_prefix}:probe:{i}')
                row.update(token_index=j, top20_logprobs=cast(Json, dict(top)), probe_prefix_token_ids=cast(Json, prefix),
                           reference_token_id=ref[j], experiment_token_id=out[j], **near_tie(top, ref[j], out[j], threshold))
                ck.require(row['near_tie'] is True, f'greedy divergence on band prompt {i} ({len(prompt)} tokens) '
                                                    f'is not a {threshold:.3f}-nat near-tie')
        rows.append(row)
    identical = sum(1 for r in rows if record(r).get('identical') is True)
    return {'checks': rows, 'identical': identical if reference is not None else None,
            'near_tie': len(rows) - identical if reference is not None else None, 'threshold_nats': threshold}


def references(result: Mapping[str, Json]) -> list[list[int]]:
    return [token_ids(record(r)['output_token_ids']) for r in ck.rows(result['checks'])]


# --- engagement (0064's own once-per-process log line) ----------------------------------------------------------------

SPLIT_LOG = 'rows (dsv41 patch 0064, VLLM_EXL3_MOE_SPLIT_MAX_TOKENS=63)'


def split_engaged(log_text: str) -> int | None:
    """Rows of the first split call from a rank's container log, or None when 0064 never ran on that rank."""
    for line in log_text.splitlines():
        if SPLIT_LOG in line and 'EXL3 routed experts: a ' in line:
            m = int(line.split('EXL3 routed experts: a ', 1)[1].split('-row call', 1)[0])
            ck.require(in_band(m), f'0064 split logged for a {m}-row call outside {BAND}')
            return m
    return None


# --- band cells ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BandCell:
    name: str
    decoders: tuple[str, ...]           # window_cells.PROMPTS keys
    max_tokens: tuple[int, ...]         # per decoder; ignore_eos: fixed work
    staged: bool                        # True: decoder 0 alone first (B1 chunks), the others once it decodes
    estimate_s: float

    def __post_init__(self) -> None:
        if len(self.decoders) != DECODERS or len(self.max_tokens) != DECODERS:
            raise ValueError(f'{self.name}: exactly {DECODERS} decoders')

    @property
    def archive_max_tokens(self) -> int:
        """The cell definition's max_tokens for the control archive (decision_v2.archive_append): the largest decoder."""
        return max(self.max_tokens)


# ESTIMATED from MEASURED NBW-W3 controls (offload/up3/budget.json @ac23d4f): 32K B3 TTFT <= 51.7 s, 1,500 tokens in
# ~59 s at B3 (~25 tok/s); 150K B1 TTFT 69 s. 1,200 decoder tokens at ~20 tok/s beside the short prompts ~ 60 s.
BAND_CELLS = {c.name: c for c in (
    BandCell('band-32k-4user', ('32k-1', '32k-2', '32k-3'), (1200, 1200, 1200), staged=False, estimate_s=150),
    # the 150K document first (B1, 4,096-row chunks, MEASURED 69 s), then two 32K requests beside its decode (ESTIMATED
    # ~25 s at 1,024-row chunks), then the short prompts beside three decoders at 150K/32K context
    BandCell('band-150k-4user', ('150k', '32k-1', '32k-2'), (1800, 1200, 1200), staged=True, estimate_s=210))}


@dataclass
class Plan:
    deadline: float                     # wall clock (time.time) submission deadline
    salt_prefix: str


def metric_delta(before: Mapping[str, float], after: Mapping[str, float], name: str) -> float:
    keys = {k for k in before.keys() | after.keys() if k.split('{')[0] == name}
    ck.require(bool(keys) and all(k in after and after[k] >= before.get(k, 0) for k in keys),
               f'missing/reset server counter {name}')
    return sum(after[k] - before.get(k, 0) for k in keys)


def run_band_cell(api: Api, counters: Counters, cell: BandCell, texts: Mapping[str, tuple[str, int]],
                  tiny: Sequence[int], plan: Plan, *, clock: Callable[[], float] = time.monotonic,
                  wall: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                  settle_s: float = 2.0) -> dict[str, Json]:
    """One repeat of a band cell. Numbers and hashes only; refuses if the band census is below the registration."""
    if wall() + cell.estimate_s > plan.deadline:
        raise RuntimeError(f'window submission deadline reached: {cell.name} does not fit')
    model = api.model()
    ck.require(model.get('max_model_len') == wc.CAP, 'served cap differs from production 327,680')
    ck.require(wc.busy(api.metrics()) == 0, 'serve not verified idle before the band cell')
    before = counters.parse_metrics(api.metrics())
    origin = clock()
    firsts = [threading.Event() for _ in cell.decoders]

    def decoder(i: int) -> dict[str, Json]:
        key = cell.decoders[i]
        text, tokens = texts[key]
        salt = wc.salt(plan.salt_prefix, cell.name, i)
        body: dict[str, Json] = {'model': model['id'], 'messages': [{'role': 'user', 'content': text}],
                                 'max_tokens': cell.max_tokens[i], 'temperature': 1.0, 'top_p': 1.0, 'seed': 42,
                                 'priority': 10, 'cache_salt': salt, 'chat_template_kwargs': TEMPLATE,
                                 'ignore_eos': True, 'stream': True, 'stream_options': {'include_usage': True}}
        sent = clock() - origin
        summary = api.stream(body, firsts[i])
        usage = record(summary['usage'])
        cached = record(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        ck.require(usage.get('prompt_tokens') == tokens and cached is not None and number(cached) == 0,
                   'decoder usage differs from the built prompt or a fresh salt hit the cache')
        ck.require(usage.get('completion_tokens') == cell.max_tokens[i] and summary.get('finish') == 'length',
                   'decoder did not generate its fixed length')
        return {'stream': i, 'prompt': key, 'prompt_tokens': tokens, 'sent_rel_s': sent, 'done_rel_s': clock() - origin,
                'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'cache_salt_sha256': hashlib.sha256(salt.encode()).hexdigest(),
                'ttft_s': summary['ttft_s'], 'usage': usage, 'finish': summary['finish']}

    def wait_first(indices: Sequence[int], futures: Sequence[Future[dict[str, Json]]]) -> None:
        end = clock() + TTFT_TIMEOUT_S
        while not all(firsts[i].is_set() for i in indices):
            for f in futures:
                if f.done():
                    f.result()
                    raise RuntimeError('a decoder finished before every decoder reached decode')
            ck.require(clock() < end, 'decoders did not all reach decode before the TTFT bound')
            sleep(.02)

    tiny_rows_out: list[Json] = []
    with ThreadPoolExecutor(max_workers=DECODERS) as pool:
        futures: list[Future[dict[str, Json]]] = []
        try:
            if cell.staged:
                futures.append(pool.submit(decoder, 0))
                wait_first([0], futures)
                futures.extend(pool.submit(decoder, i) for i in range(1, DECODERS))
            else:
                futures.extend(pool.submit(decoder, i) for i in range(DECODERS))
            wait_first(range(DECODERS), futures)
            band_start = clock() - origin
            j = 0
            while not any(f.done() for f in futures) and j < MAX_TINY and wall() < plan.deadline:
                prompt = tiny_prompt(tiny, j)
                salt = f'{plan.salt_prefix}:{cell.name}:tiny:{j}'
                sent = clock()
                r = api.post('/v1/completions', {'model': MODEL, 'prompt': cast(Json, prompt), 'max_tokens': 1,
                                                 'temperature': 0, 'priority': 10, 'cache_salt': salt})
                done = clock()
                usage = record(r['usage'])
                cached = record(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
                beside = sum(1 for f in futures if not f.done())
                tiny_rows_out.append({'j': j, 'length': len(prompt), 'prompt_tokens': usage.get('prompt_tokens'),
                                      'cached_tokens': cached, 'completion_tokens': usage.get('completion_tokens'),
                                      'latency_s': done - sent, 'beside': beside})
                j += 1
            band_end = clock() - origin
            decoders = [f.result() for f in futures]
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    sleep(settle_s)                     # counter publication, outside every measured request
    after = counters.parse_metrics(api.metrics())
    rounds = metric_delta(before, after, 'vllm:spec_decode_num_drafts_total')
    accepted = metric_delta(before, after, 'vllm:spec_decode_num_accepted_tokens_total')
    seconds = metric_delta(before, after, 'vllm:request_decode_time_seconds_sum')
    ck.require(rounds > 0 and seconds > 0, 'no spec rounds/server decode time')
    census = census_of(cast(list[Mapping[str, Json]], tiny_rows_out), rounds)
    result: dict[str, Json] = {
        'cell': cell.name, 'max_tokens': cell.archive_max_tokens, 'decoders': cast(Json, decoders),
        'band_start_rel_s': band_start, 'band_end_rel_s': band_end, 'tiny': tiny_rows_out, **census,
        'ms_per_round': 1000 * seconds / rounds, 'accepted_per_round': accepted / rounds, 'spec_request_rounds': rounds,
        'decoder_ttft_s': cast(Json, [record(d)['ttft_s'] for d in decoders]),
        'round_time_scope': 'server request-decode duration sum / spec request-round count (nbw_cells.summarize); the '
                            'short prompts add no decode time and no rounds; includes every band step',
        'status': 'MEASURED unprofiled client timing and server counters'}
    ck.require(result['band_ok'] is True, f'{cell.name}: band census {census["band_steps"]} steps / fraction '
                                           f'{census["band_fraction"]:.3f} below the registration; cannot decide 0064')
    return result


def census_of(rows: Sequence[Mapping[str, Json]], rounds: float) -> dict[str, Json]:
    """Band steps = short prompts with their exact registered length, no cache hit, one output token, prefilled while
    every decoder was alive (their step had length + 3..15 rows, inside 33..63)."""
    valid = [r for r in rows if r.get('prompt_tokens') == r.get('length') and r.get('cached_tokens') == 0
             and r.get('completion_tokens') == 1]
    ck.require(len(valid) == len(rows), 'a short prompt was cached, truncated, lacked cached_tokens or produced more '
                                        'than one token')
    band = [r for r in valid if r.get('beside') == DECODERS and int(number(r['length'])) in TINY_LENGTHS
            and all(in_band(x) for x in tiny_rows(int(number(r['length']))))]
    per_decoder_rounds = rounds / DECODERS
    fraction = len(band) / per_decoder_rounds if per_decoder_rounds > 0 else 0.0
    latencies = [number(r['latency_s']) for r in valid]
    return {'short_prompts': len(rows), 'band_steps': len(band), 'band_fraction': fraction,
            'short_latency_median_s': statistics.median(latencies) if latencies else None,
            'band_ok': len(band) >= MIN_BAND_STEPS and fraction >= MIN_BAND_FRACTION,
            'band_registration': f'>= {MIN_BAND_STEPS} band steps and >= {MIN_BAND_FRACTION:.0%} of decoder rounds'}
