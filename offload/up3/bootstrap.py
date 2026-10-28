"""In-worker lifecycle for the coordinator's qualification-only integration.

No worker injection, dev HTTP endpoint, pickle, SSH, Docker mutation or model
launcher is hidden here. The caller supplies the *existing* worker on both ranks.
before_init must precede init_device/model loading for native TID attribution.
after_warmup runs synchronous native tests on both ranks; K5 remains installed
only across the coordinator's public B3/B4 requests. finish always removes hooks.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from offload.prefill_maxlen.evidence import Json
from offload.up3.engine_common import Cases, require
from offload.up3.qualify import MARKER, Request, engine_sources, receipt, worker_identity

if TYPE_CHECKING:
    import torch
    from offload.up3.engine_affinity import Births
    from offload.up3.engine_k5 import Observer


class Session:
    def __init__(self, worker: object, request: Request):
        request.validate()
        self.worker, self.request = worker, request
        self.births: Births | None = None
        self.observer: Observer | None = None
        self.cases: Cases = {}
        self.begun = self.warmed = self.finished = False

    def before_init(self) -> None:
        require(not self.begun, 'bootstrap reused')
        engine_sources(self.request.arm)
        self.begun = True
        if self.request.arm == 'cpu-pin':
            from offload.up3.engine_affinity import Births
            self.births = Births()
            self.births.install()

    def after_warmup(self, *, rank_log: str = '') -> None:
        require(self.begun and not self.warmed and not self.finished, 'bootstrap lifecycle/order differs')
        self.warmed = True
        arm = self.request.arm
        try:
            worker_identity(self.request, self.worker)
            if arm == 'split':
                from offload.up3.engine_split import run
                # Cannot emit until actual teacher-forced logits and task review are supplied.
                self.cases = run(self.worker, rounding_evidence={})
            elif arm == 'vor':
                from offload.up3.engine_vor import run as vor
                self.cases = vor(self.worker)
            elif arm == 'k5':
                from offload.up3.engine_k5 import Observer, native_gate
                self.cases['noncausal-stock'] = native_gate(self.worker)
                self.observer = Observer(self.worker)
                self.observer.install()
            elif arm == 'dual-ring':
                from offload.up3.engine_ring import run as ring
                self.cases = ring(self.worker, rank_log=rank_log)
            else:
                require(self.births is not None, 'pin bootstrap missing early TID evidence')
        except BaseException:
            self.close()
            raise

    def finish(self, *, logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
               task_review: dict[str, Json] | None = None, emit: bool = True) -> dict[str, Json]:
        require(self.warmed and not self.finished, 'bootstrap finish without completed native phase/reused')
        self.finished = True
        try:
            if self.request.arm == 'split':
                from offload.up3.engine_split import rounding
                require(logits is not None and task_review is not None, 'actual C/C/A teacher-forced logits and bound task review required')
                assert logits is not None and task_review is not None
                require(all(value.is_cuda for value in logits), 'actual worker CUDA logits required, not CPU placeholders')
                self.cases['teacher-forced-rounding'] = rounding(*logits, task_review, run=self.request.run)
            elif self.request.arm == 'k5':
                require(self.observer is not None, 'actual K5 observer missing')
                assert self.observer is not None
                self.cases = self.observer.collect(self.cases['noncausal-stock'])
            elif self.request.arm == 'cpu-pin':
                from offload.up3.engine_affinity import run
                require(self.births is not None, 'actual native births missing')
                assert self.births is not None
                self.cases = run(self.worker, self.births)
            result = receipt(self.request, self.cases, self.worker)
            if emit:
                print(MARKER+json.dumps(result, allow_nan=False), flush=True)
            return result
        finally:
            self.close()

    def close(self) -> None:
        if self.observer is not None:
            self.observer.close()
        if self.births is not None:
            self.births.close()
