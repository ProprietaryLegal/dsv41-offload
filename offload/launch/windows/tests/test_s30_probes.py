"""S30 independent-check probes for the UP2 low-cap runner (offline; no Spark, x86host, GPU or HTTP server).

Each FIX-n probe drives the runner's real Live code with outputs shaped like the deployed system's. At
3490b1f/76ace38 they were strict xfails against the unmodified 07abaa8 runner (each raised its STOP); with
the S30 suggested fixes they pass. The control tests pass either way.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from offload.prefill_maxlen.evidence import Json
from offload.launch.windows import up2_checks as ck
from offload.launch.windows.tests.test_up2_low import StartFake

# Rank 0 (sparkhead) runs the API server and EngineCore; only EngineCore logs the KV capacity
# (src-image vllm/v1/engine/core.py:353 -> kv_cache_utils.update_kv_cache_capacity).
RANK0_LOG = ('(EngineCore pid=304) INFO [kv_cache_utils.py:2507] GPU KV cache size: 2,502,600 tokens, '
             'Maximum concurrency for 327,680 tokens per request: 7.64x\n'
             '(Worker_TP0 pid=376) INFO [engram.py:1189] Engram table rows from the remote table server '
             'rdma://10.202.28.1:18610 (identity x); the local shard y is the exact fallback\n'
             '(Worker_TP0 pid=376) INFO remote DSpark (capacity policy) rank 0 -> o1rdma://10.202.28.1:18620 '
             'through proxy process pid 595; rank-0 ...\n'
             '(Worker_TP0 pid=376) INFO remote DSpark async mode on (host-flag hand-off self-test passed; x)\n')
# Rank 1 (sparkworker, launched with --headless by dsv41_tp2_offload.sh) runs only MultiprocExecutor workers
# (src-image vllm/entrypoints/cli/serve.py run_headless): no EngineCore, so no KV-capacity line.
RANK1_LOG = ('INFO [serve.py:224] Launching vLLM (v0.31.1) headless multiproc executor, with head node address '
             '10.202.26.2:25000 for torch.distributed process group.\n'
             '(Worker_TP1 pid=210) INFO [engram.py:1189] Engram table rows from the remote table server '
             'rdma://10.202.29.1:18610 (identity x); the local shard y is the exact fallback\n')


def proxy_stats(*, fires: int, rounds_ok: int, last_failed_round: int = 0) -> dict[str, Json]:
    """Keys/values as in a real K1 stats file (K1-production/stats-post.json, W8-native/stats-C2.json)."""
    return {'counters':{'rounds_ok':rounds_ok,'rounds_failed':int(fires > 0),'watchdog_preempted':fires},
            'policy':'capacity','async':True,'async_dead':None,'proxy_torch_loaded':False,
            'async_proxy':{'mode':'process','pid':595},'async_proxy_process':{'pid':595},
            'async_handoff':{'mlock':'ok'},
            'async_watchdog':{'running':True,'deadline_ms':200,'fires':fires,
                              'last_failed_round':last_failed_round,'mirrored':True}}


class DeployedShapeFake(StartFake):
    """StartFake, but each rank returns the logs its real process layout produces, and the stats flush,
    request driver and proxy follow a fresh start whose first drafting round misses the 200 ms watchdog."""
    def __init__(self, out: Path, *, rank1_log: str = RANK1_LOG, first_round_fire: bool = False):
        super().__init__(out,'')
        self.rank1_log = rank1_log
        self.first_round_fire = first_round_fire
        self.requests = 0
        self.data_dir = out
        self.before = None
        self.last_stats = {}

    def ssh(self, host: str, command: str, stdin: str | None = None) -> str:
        if 'docker logs' in command:
            return RANK0_LOG if host == 'sparkhead' else self.rank1_log
        if '--pid' in command:
            if self.requests == 0:
                return json.dumps(proxy_stats(fires=0,rounds_ok=0))
            fired = int(self.first_round_fire)
            return json.dumps(proxy_stats(fires=fired,rounds_ok=40*self.requests,last_failed_round=fired))
        return super().ssh(host,command,stdin)

    def call(self, label: str, command: list[str], stdin: str | None = None) -> str:
        if 'offload.launch.windows.up2_request' not in command:
            return super().call(label,command,stdin)
        self.requests += 1
        output = Path(command[command.index('--output')+1])
        target = int(command[command.index('--target')+1])
        budget = int(command[command.index('--budget')+1])
        output.write_text(json.dumps({'type':'request','kind':'warm32','rep':0,'target_ctx':target,'cap':ck.CAP,
            'usage':{'prompt_tokens':target,'completion_tokens':budget},'finish':'length','cached_tokens':0})+'\n')
        return ''


def test_control_rank1_with_kv_line_starts(tmp_path: Path) -> None:
    """Control: the same fake passes when rank 1 also logs the KV line (the shape the existing fake assumes)."""
    backend = DeployedShapeFake(tmp_path,rank1_log=RANK0_LOG)
    backend.start('diag-off',0,1)
    assert backend.ids == ['rank0','rank1'] and backend.proxy_pid == 595


# S30 FIX-1: passes with the suggested fix
def test_fix1_headless_rank1_start_is_accepted(tmp_path: Path) -> None:
    backend = DeployedShapeFake(tmp_path)
    backend.start('diag-off',0,1)                       # 07abaa8: STOP: rank KV capacity mismatch
    assert backend.ids == ['rank0','rank1']


def test_control_warmup_without_watchdog_fire_passes(tmp_path: Path) -> None:
    backend = DeployedShapeFake(tmp_path,rank1_log=RANK0_LOG)
    backend.start('diag-off',0,1)
    backend.request('diag-off','warm32',32000,32,force_length=True)
    backend.clean('diag-off',1)                         # end-of-arm check, zero baseline (SCREEN: no native receipt)


# S30 FIX-2: passes with the suggested fix
def test_fix2_first_round_watchdog_fire_is_recorded_not_stopped(tmp_path: Path) -> None:
    backend = DeployedShapeFake(tmp_path,rank1_log=RANK0_LOG,first_round_fire=True)
    backend.start('diag-off',0,1)
    backend.request('diag-off','warm32',32000,32,force_length=True)   # 07abaa8: STOP: new watchdog fires
    backend.clean('diag-off',1)
    rows = [json.loads(line) for line in (tmp_path/'proxy-stats.jsonl').read_text().splitlines()]
    assert rows[-1]['async_watchdog']['fires'] == 1 and rows[-1]['async_watchdog']['last_failed_round'] == 1


def swap_trajectory_monitor(tmp_path: Path, swap_free_kb: list[int]) -> Path:
    """A local stand-in for the per-host /proc/meminfo stream (no SSH). MemAvailable stays well above 3 GiB."""
    script = tmp_path/'meminfo-fake.py'
    lines = ''.join(f'print("MemAvailable: 6773784 kB;SwapFree: {kb} kB",flush=True)\ntime.sleep(.05)\n'
                    for kb in swap_free_kb)
    script.write_text('import time\n'+lines+'time.sleep(10)\n')
    return script


# MEASURED sparkhead SwapFree (kB), W10 R6 memory logs (window-smoke-r6-20261024T203207Z): production serving,
# production stopped by start_exp, then the experiment model load. W8 mem-*.log show the same ~4.3 GiB swap-out at
# every start on both Sparks; MemAvailable never fell below 4.8 GiB. Every runner that ran clean (W8, W9, W10) gates
# MemAvailable only.
W10_R6_SPARKHEAD_SWAP_FREE_KB = [12343640, 15791000, 11070804]


# S30 FIX-3: passes with the suggested fix
def test_fix3_start_time_swap_out_is_not_new_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from offload.launch.windows import up2_low as low
    script = swap_trajectory_monitor(tmp_path,W10_R6_SPARKHEAD_SWAP_FREE_KB)
    monkeypatch.setattr(low,'SSH',[sys.executable,str(script)])
    supervisor = low.Supervisor(tmp_path,__import__('time').time()+60)
    try:
        supervisor.begin()
        supervisor.run('owned-start-exp',[sys.executable,'-c','import time; time.sleep(.5)'])   # 07abaa8: STOP: new swap
    finally:
        supervisor.close()
    recorded = [json.loads(line)['swap_free_kb'] for line in (tmp_path/'memory-sparkhead.jsonl').read_text().splitlines()]
    assert recorded == W10_R6_SPARKHEAD_SWAP_FREE_KB                    # the swap-out stays a record


def test_control_non_decreasing_swap_free_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from offload.launch.windows import up2_low as low
    script = swap_trajectory_monitor(tmp_path,[12343640,15791000,15791000])
    monkeypatch.setattr(low,'SSH',[sys.executable,str(script)])
    supervisor = low.Supervisor(tmp_path,__import__('time').time()+60)
    try:
        supervisor.begin()
        supervisor.run('owned-start-exp',[sys.executable,'-c','import time; time.sleep(.5)'])
    finally:
        supervisor.close()
