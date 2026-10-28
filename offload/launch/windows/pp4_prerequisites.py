"""Local PP4 admission checks. No host contact; gate evidence is explicit and hash-bound."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

from offload.launch.windows import pf1_window as pf
from offload.launch.windows import up2_checks as ck
from offload.prefill_maxlen.evidence import Json, record

GATES = ('whole_k2_tp2', 'dyadic24_acceptance', 't0_n4_identity', 'graph_sampled',
         'positive_lower_bound_cost_margin', 'both_rank_q_failure_target_only',
         'grammar_penalty_stop_eos', 'tool_json_reasoning_parser', 'cancel_reset_reconnect_timeout_late_reply',
         'gb10_image_and_abi_reconciliation', 'complete_round_instrumentation', 'independent_review')
# Round5 coordinator decision: calibration is retired, not silently passed.
RETIRED_GATES: dict[str, Json] = {'measured_complete_block_budget':
    'Round5 replaces two full calibration blocks with a separately registered historical-input budget; not a qualification pass'}
OVERLAY_REL = 'offload/launch/windows/pp4-staged/verifier'
RUNTIME = '/opt/dsv41/x86host-data/pp4/source/runtime'
PYTHON = '/home/operator/miniconda3/bin/python3'
STOCK = 'stock/exllamav3_ext.cpython-313-x86_64-linux-gnu.so'
FACC = 'facc/exllamav3_ext_facc.cpython-313-x86_64-linux-gnu.so'


def runtime_artifacts(data: dict[str, Json]) -> dict[str, Json]:
    """Never substitute production CPython3.12 binaries for the required PP4 CPython3.13 runtime."""
    key = pf.digest(pf.ROOT / 'offload/dspark/worker/p2p/p2p.cu')[:16]
    artifacts = record(data['runtime_artifact_sha256'])
    ck.require(set(artifacts) == {STOCK, FACC, f'p2p/{key}/dspark_p2p.so'}, 'three reviewed CPython3.13 runtime binaries required')
    ck.require(all(re.fullmatch('[0-9a-f]{64}', str(v)) for v in artifacts.values()), 'runtime artifact digest missing')
    ck.require(data.get('runtime_python') == PYTHON and data.get('runtime_torch') == '2.12.1+cu130',
               'required interpreter/Torch runtime differs')
    return artifacts


def staging(path: Path, expected: str) -> dict[str, Json]:
    data = record(cast(Json, json.loads(path.read_text())))
    ck.require(data.get('source_commit') == expected and data.get('staging_verdict') == 'REVIEWED',
               'independently reviewed source/runtime staging receipt required')
    runtime_artifacts(data)
    sources = record(data['runtime_sources'])
    ck.require(set(sources) == {'stock', 'facc', 'p2p'}, 'three private X86host runtime source roots required')
    ck.require(all(re.fullmatch(r'/opt/dsv41/x86host-data/pp4/[A-Za-z0-9_./-]+', str(p))
                   and '..' not in str(p).split('/') for p in sources.values()), 'runtime source outside authorized X86host tree')
    return data


def qualification(path: Path, expected: str) -> dict[str, Json]:
    data = record(cast(Json, json.loads(path.read_text())))
    ck.require(data.get('schema') == 'pp4-live-admission-v1' and data.get('source_commit') == expected,
               'PP4 qualification receipt schema/source mismatch')
    ck.require(data.get('verdict') == 'QUALIFIED', 'PP4 qualifications are not complete; no window may open')
    runtime_artifacts(data)
    gates = record(data['gates'])
    ck.require(set(gates) == set(GATES), 'missing/extra PP4 qualification gate')
    for name in GATES:
        row = record(gates[name])
        ck.require(row.get('pass') is True and isinstance(row.get('evidence_path'), str), f'{name}: qualification absent')
        evidence = Path(str(row['evidence_path']))
        ck.require(evidence.is_absolute() and evidence.is_file(), f'{name}: primary evidence file absent')
        ck.require(re.fullmatch('[0-9a-f]{64}', str(row.get('sha256'))) is not None
                   and pf.digest(evidence) == row['sha256'], f'{name}: primary evidence changed')
    from offload.launch.windows.pp4_budget import admitted
    data['registered_budget'] = admitted(path, expected)
    data['block_budget_s'] = record(data['registered_budget'])['block_budget_s']
    ck.require(data.get('depth') == 5, 'this preregistration is fixed5; fixed4 needs a new registration')
    ck.require(all(re.fullmatch('[0-9a-f]{64}', str(data.get(k))) for k in ('worker_identity', 'markov_identity')),
               'reviewed K2 checkpoint identities required')
    outputs = record(data['verifier_sha256'])
    from offload.dspark.probabilistic.pp3.prepare_verifier import PINS
    from offload.launch.windows.pp4_prepare import HELPER
    ck.require(set(outputs) == {*PINS, HELPER} and all(re.fullmatch('[0-9a-f]{64}', str(x)) for x in outputs.values()),
               'prepared verifier, failure-mask and round-helper digests required')
    return data


def overlay_pins(receipt: dict[str, Json]) -> dict[str, Json]:
    remote = pf.ROOT / 'offload/dspark/spark/remote_dspark'
    pins: dict[str, Json] = {pf.DIST + 'vllm/v1/worker/gpu/spec_decode/remote_dspark/' + p.name: pf.digest(p) for p in sorted(remote.glob('*.py'))}
    pins.update({pf.DIST + name: value for name, value in record(receipt['verifier_sha256']).items()})
    return pins
