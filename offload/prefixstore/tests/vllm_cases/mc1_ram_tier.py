"""Test stand-in for a later Tier 2 (e.g. x86host RAM): a matter-gated secondary tier with no file I/O at all.

It exists to prove the connector is tier-agnostic. It implements only the persistent-tier contract documented in
``dsv41_stream`` (``dsv41_matter_gated``, ``allows_get``, the MatterGate from ``tiering/fs/dsv41_matter.py``) and
the SecondaryTierManager API, and the unchanged connector stores, streams and refuses through it exactly as
through Tier 1. Its data lives in a process-wide dict per ``store`` name, so it outlives an engine restart inside
a test, as a remote store would. Copies run inline (a real Tier 2 would queue them); results are reported through
``get_finished_jobs`` like any tier's."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

from vllm.v1.kv_offload.base import Locality, LookupResult, Medium, OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    RequestOffloadingContext,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.fs import dsv41_matter as matter

STORES: dict[str, dict[bytes, bytes]] = {}


@dataclass
class _ReqState:
    tag: matter.RequestTag
    audited: bool = False


class RamMatterTier(SecondaryTierManager):  # type: ignore[misc]
    medium: ClassVar[Medium] = Medium.CPU
    dsv41_matter_gated: ClassVar[bool] = True

    def __init__(self, offloading_spec: Any, primary_kv_view: memoryview, tier_type: str, store: str,
                 excluded_ns_file: str, excluded_ns_sha256: str, audit_path: str,
                 backpressure_detector: Any = None) -> None:
        super().__init__(offloading_spec, primary_kv_view, tier_type, backpressure_detector)
        self.locality = Locality.REMOTE
        self._flat = primary_kv_view.cast('B')
        self._data = STORES.setdefault(store, {})
        rank = int(offloading_spec.config.parallel.rank)
        self.gate = matter.MatterGate(matter.ExclusionSet(excluded_ns_file, excluded_ns_sha256),
                                      matter.AuditLog(audit_path, rank))
        self._done: list[JobResult] = []

    def _decide(self, ctx: ReqContext) -> tuple[matter.Verdict, str | None]:
        state: _ReqState | None = ctx.get_state(_ReqState)
        if state is None:
            state = _ReqState(self.gate.classify(ctx.kv_transfer_params))
            ctx.set_state(state)
        verdict = self.gate.decide(state.tag)
        if verdict in matter.AUDITED_REFUSALS and not state.audited and self.gate.audit is not None:
            state.audited = True
            self.gate.audit.write('refuse', state.tag.ns, reason=verdict.value, tier=self.tier_type)
        return verdict, state.tag.ns

    def _slot(self, chunk_id: int) -> slice:
        return slice(int(chunk_id) * self.block_size_bytes, (int(chunk_id) + 1) * self.block_size_bytes)

    def allows_get(self, req_context: ReqContext) -> bool:
        return self._decide(req_context)[0] is matter.Verdict.ALLOW

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self._decide(req_context)
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        verdict, ns = self._decide(req_context)
        if verdict is not matter.Verdict.ALLOW or ns is None:
            return LookupResult.MISS
        return LookupResult.HIT if bytes.fromhex(ns) + key in self._data else LookupResult.MISS

    def submit_store(self, job_metadata: TransferJob) -> None:
        verdict, ns = self._decide(job_metadata.req_context)
        if verdict is not matter.Verdict.ALLOW or ns is None:
            self._done.append(JobResult(job_id=job_metadata.job_id, success=False, transfer_bytes=0))
            return
        for key, cid in zip(job_metadata.keys, job_metadata.chunk_ids):
            self._data[bytes.fromhex(ns) + key] = bytes(self._flat[self._slot(cid)])
        nbytes = len(job_metadata.keys) * self.block_size_bytes
        assert self.gate.audit is not None
        self.gate.audit.write('put', ns, chunks=len(job_metadata.keys), bytes=nbytes, tier=self.tier_type)
        self._done.append(JobResult(job_id=job_metadata.job_id, success=True, transfer_bytes=nbytes))

    def submit_load(self, job_metadata: TransferJob) -> None:
        verdict, ns = self._decide(job_metadata.req_context)
        keys = list(job_metadata.keys)
        if verdict is not matter.Verdict.ALLOW or ns is None:
            self._done.append(JobResult(job_id=job_metadata.job_id, success=False, transfer_bytes=0))
            return
        done = 0
        for key, cid in zip(keys, job_metadata.chunk_ids):
            blob = self._data.get(bytes.fromhex(ns) + key)
            if blob is None:
                break
            self._flat[self._slot(cid)] = blob
            done += 1
        assert self.gate.audit is not None
        if done:
            self.gate.audit.write('get-hit', ns, chunks=done, bytes=done * self.block_size_bytes, tier=self.tier_type)
        ok = done == len(keys)
        self._done.append(JobResult(job_id=job_metadata.job_id, success=ok,
                                    successful_keys=tuple(keys[:done]) if not ok and done else None,
                                    transfer_bytes=done * self.block_size_bytes))

    def get_finished_jobs(self) -> Iterable[JobResult]:
        out, self._done = self._done, []
        return out

    def drain_jobs(self) -> None:
        """Copies are inline: nothing is ever in flight."""

    def entries(self, ns: str) -> int:
        prefix = bytes.fromhex(ns)
        return sum(1 for k in self._data if k.startswith(prefix))
