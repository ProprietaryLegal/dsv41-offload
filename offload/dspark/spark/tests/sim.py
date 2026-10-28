# SPDX-License-Identifier: Apache-2.0
"""A small simulator of the engine side of DSpark speculation, for offline tests.

It produces, step by step, what the model runner hands to ``propose``: an InputBatch-like
object (batch order, query_start_loc_np, positions, slots, prefill flags), the three tap
tensors, num_sampled / num_rejected, last_sampled and next_prefill_tokens. It reproduces
the properties the contract relies on:
  * chunked prefill, with chunks shorter and longer than the 128 window;
  * decoder bounded replay: in a prefill step only the last min(q, 128) rows of a request
    have real taps, the others are zeros;
  * decode verify with a per-request k (0..5) and random rejections; rejected rows carry
    garbage taps (they must never reach the worker);
  * slot reuse by new requests, re-add of the same request at a lower position
    (preemption + prefix replay), new requests starting at a replay position;
  * requests not scheduled in some steps.
``truth`` keeps the taps the target computed for every committed position, so a test can
compute the drafts the worker must return.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

W = 128
H = 4
N_TAPS = 3
N = 5
M = 4
VOCAB = 129280


def tap_rows(doc: int, positions: np.ndarray, salt: int = 0) -> torch.Tensor:
    """[len, N_TAPS*H] bf16, exactly representable values keyed by (doc, pos)."""
    p = torch.as_tensor(positions, dtype=torch.int64).view(-1, 1)
    j = torch.arange(N_TAPS * H, dtype=torch.int64).view(1, -1)
    v = (doc * 131 + p * 17 + j * 3 + salt * 7) % 64
    return (v.to(torch.float32) / 8.0).to(torch.bfloat16)


def row_bytes(row: torch.Tensor) -> bytes:
    return row.contiguous().view(torch.uint8).numpy().tobytes()


@dataclass
class SimRequest:
    req_id: str
    doc: int
    slot: int
    computed: int           # next position the target processes (= anchor position)
    prefill_end: int        # position where the prompt (or re-add replay) ends
    start: int              # first position with rows in this lifetime
    decode_left: int
    anchor: int = 0
    truth: dict[int, bytes] = field(default_factory=dict)

    @property
    def prefilling(self) -> bool:
        return self.computed < self.prefill_end


@dataclass
class FakeInputBatch:
    req_ids: list[str]
    num_reqs: int
    idx_mapping_np: np.ndarray
    idx_mapping: torch.Tensor
    query_start_loc_np: np.ndarray
    positions: torch.Tensor
    is_prefilling_np: np.ndarray
    num_computed_prefill_tokens_np: np.ndarray
    prefill_len_np: np.ndarray
    num_draft_tokens_per_req: np.ndarray | None
    num_tokens: int


@dataclass
class StepData:
    batch: FakeInputBatch
    taps: list[torch.Tensor]
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    last_sampled: torch.Tensor
    next_prefill_tokens: torch.Tensor
    reqs: list[SimRequest]          # batch order
    need_draft: list[bool]


class Engine:
    def __init__(self, seed: int = 0, chunk_sizes=(37, 64, 128, 129, 300, 1000)):
        self.rng = random.Random(seed)
        self.chunk_sizes = chunk_sizes
        self.slots: list[SimRequest | None] = [None] * M
        self.next_id = 0
        self.next_doc = 1
        self.last_sampled = torch.zeros(M, 1, dtype=torch.int64)
        self.next_prefill = torch.zeros(1, M, dtype=torch.int32)

    # ------------------------------------------------------------- lifecycle

    def new_request(self, slot: int, prompt_len: int | None = None, hit: int = 0) -> SimRequest:
        prompt_len = prompt_len or self.rng.choice([5, 90, 127, 128, 129, 700, 2100])
        start = 0
        hit = min(hit, prompt_len - 1)   # the scheduler always recomputes the last token
        if hit > W:  # prefix hit of `hit` tokens: SWA bounded replay recomputes the last W
            hit -= hit % 64
            start = hit - W
        r = SimRequest(req_id=f"req-{self.next_id}", doc=self.next_doc, slot=slot,
                       computed=start, prefill_end=prompt_len, start=start,
                       decode_left=self.rng.randint(3, 25))
        self.next_id += 1
        self.next_doc += 1
        self.slots[slot] = r
        return r

    def readd(self, slot: int) -> SimRequest:
        """Preempted and re-added (same request id): prefix hit + replay of its own tokens."""
        r = self.slots[slot]
        assert r is not None
        # its tokens now include the pending anchor (position r.computed), which the
        # re-add prefill processes before sampling the next token
        total = r.computed + 1
        hit = (total - 1) - (total - 1) % 64
        if hit <= W:
            r.start, r.computed = 0, 0
        else:
            r.start = r.computed = hit - W
        r.prefill_end = total
        # committed taps below the replay start stay valid (causal); nothing to drop
        return r

    # ------------------------------------------------------------- one step

    def step(self, schedule: list[int] | None = None) -> StepData:
        live = [i for i, r in enumerate(self.slots) if r is not None]
        if schedule is None:
            schedule = [i for i in live if self.rng.random() > 0.15] or live[:1]
        reqs = [self.slots[i] for i in schedule]
        qs, pos_list, tap_list, nrej, nsamp, need, ndraft = [], [], [], [], [], [], []
        is_pref, ncomp, plen = [], [], []
        for r in reqs:
            if r.prefilling:
                chunk = min(r.prefill_end - r.computed, self.rng.choice(self.chunk_sizes))
                p = np.arange(r.computed, r.computed + chunk)
                rows = tap_rows(r.doc, p)
                kept = min(chunk, W)
                fed = rows.clone()
                fed[: chunk - kept] = 0     # decoder bounded replay: trimmed rows are zeros
                for k, pp in enumerate(p):
                    r.truth[int(pp)] = row_bytes(rows[k])
                is_pref.append(True)
                ncomp.append(r.computed)
                plen.append(r.prefill_end)
                r.computed += chunk
                done = r.computed == r.prefill_end
                if done:
                    r.anchor = self.rng.randrange(VOCAB)
                    self.last_sampled[r.slot, 0] = r.anchor
                    nsamp.append(1)
                else:
                    r.anchor = (r.doc * 7919 + r.computed) % VOCAB  # the next prompt token
                    self.next_prefill[0, r.slot] = r.anchor
                    nsamp.append(0)
                nrej.append(0)
                need.append(done)
                ndraft.append(0)
            else:
                k = self.rng.randint(0, N)
                p = np.arange(r.computed, r.computed + 1 + k)
                rows = tap_rows(r.doc, p)
                rej = self.rng.randint(0, k)
                valid = 1 + k - rej
                fed = rows.clone()
                if rej:
                    fed[valid:] = tap_rows(r.doc, p[valid:], salt=99)   # garbage
                for kk in range(valid):
                    r.truth[int(p[kk])] = row_bytes(rows[kk])
                is_pref.append(False)
                ncomp.append(r.prefill_end)
                plen.append(r.prefill_end)
                r.computed += valid
                r.anchor = self.rng.randrange(VOCAB)
                self.last_sampled[r.slot, 0] = r.anchor
                nrej.append(rej)
                nsamp.append(valid)
                need.append(True)
                ndraft.append(k)
                r.decode_left -= 1
            qs.append(len(p))
            pos_list.append(p)
            tap_list.append(fed)
        qsl = np.zeros(len(reqs) + 1, dtype=np.int64)
        np.cumsum(qs, out=qsl[1:])
        positions = torch.as_tensor(np.concatenate(pos_list), dtype=torch.int64)
        allrows = torch.cat(tap_list, dim=0)
        taps = [allrows[:, t * H:(t + 1) * H].contiguous() for t in range(N_TAPS)]
        slots = np.array([r.slot for r in reqs], dtype=np.int64)
        batch = FakeInputBatch(
            req_ids=[r.req_id for r in reqs], num_reqs=len(reqs), idx_mapping_np=slots,
            idx_mapping=torch.as_tensor(slots, dtype=torch.int32), query_start_loc_np=qsl,
            positions=positions, is_prefilling_np=np.array(is_pref),
            num_computed_prefill_tokens_np=np.array(ncomp), prefill_len_np=np.array(plen),
            num_draft_tokens_per_req=np.array(ndraft, dtype=np.int32),
            num_tokens=int(qsl[-1]))
        return StepData(batch=batch, taps=taps,
                        num_sampled=torch.tensor(nsamp, dtype=torch.int32),
                        num_rejected=torch.tensor(nrej, dtype=torch.int32),
                        last_sampled=self.last_sampled.clone(),
                        next_prefill_tokens=self.next_prefill.clone(),
                        reqs=reqs, need_draft=need)

    def expected_window(self, r: SimRequest) -> list[tuple[int, bytes]]:
        a = r.computed
        return [(p, r.truth[p]) for p in range(max(a - W, r.start), a)]
