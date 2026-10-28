"""Dual-ring qualification on the existing worker's actual TP communicator.

Both ranks MUST call together. Never instantiate a replacement ring, use NCCL
as a successful fallback, open a listener, or change a GID/device/CPU setting.
"""
from __future__ import annotations

import importlib
import ipaddress
import re
from pathlib import Path
from typing import cast

import torch

from offload.prefill_maxlen.evidence import Json
from offload.up3.engine_common import Cases, attr, invoke, passed, require, runner
from offload.up3.engine_split import equal_bytes

DEVICES = ('rocep1s0f1', 'roceP2p1s0f1')


def gid(root: Path, device: str, index: int) -> Json:
    port = root/device/'ports/1'
    address = ipaddress.IPv6Address((port/f'gids/{index}').read_text().strip())
    ipv4 = address.ipv4_mapped
    require((port/f'gid_attrs/types/{index}').read_text().strip() == 'RoCE v2' and ipv4 is not None,
            device+': GID is not IPv4 RoCE v2')
    network = ipaddress.IPv4Network('10.202.26.0/24' if device == DEVICES[0] else '10.202.27.0/24')
    require(ipv4 in network, device+': wrong fabric IPv4 GID')
    ndev = (port/f'gid_attrs/ndevs/{index}').read_text().strip()
    require(bool(ndev), device+': GID has no netdev')
    return passed(1, device=device, gid_index=index, ipv4_roce_v2=True, netdev=ndev, ipv4=str(ipv4))


def counters(root: Path) -> dict[str, int]:
    # IB port_{xmit,rcv}_data counts 4-byte units, unlike netdev byte counters.
    result = {}
    for device in DEVICES:
        port = root/device/'ports/1/counters'
        result[device] = sum(int((port/name).read_text().strip())*4 for name in ('port_xmit_data', 'port_rcv_data'))
    return result


def log_line(log: str, rank: int) -> str:
    require('NCCL stays' not in log and 'RDMA all-reduce off' not in log, 'NCCL-only fallback in rank log')
    prefix = f'EXL3 RDMA all-reduce: rank {rank} of 2 on '+','.join(DEVICES)
    lines = [line[line.index(prefix):] for line in log.splitlines() if prefix in line]
    require(len(lines) == 1 and re.search(r', gid 3, proxy on cpu \d+,', lines[0]) is not None,
            'missing/ambiguous exact current-rank dual-ring log/GID')
    return lines[0]


def collective(comm: object, ring: object, rank: int, hidden: int, device: torch.device) -> tuple[int, int]:
    require(attr(ring, 'rank') == rank and attr(ring, 'world') == 2, 'actual ring rank/world mismatch')
    count, compared = 0, 0
    for n in (8, 6*hidden, 24*hidden):
        base = (torch.arange(n, device=device) % 17).to(torch.bfloat16)
        values = base+rank
        require(invoke(attr(ring, 'fits'), values) is True, 'target-shaped all-reduce does not fit actual ring')
        actual = cast(torch.Tensor, invoke(attr(comm, 'all_reduce'), values))
        compared += equal_bytes(actual, base*2+1, 'dual-ring all-reduce')
        count += 1
    for shape in ((24, 2), (24, hidden//2)):
        for dim in (0, 1):
            base_i = torch.arange(shape[0]*shape[1], dtype=torch.int32, device=device).reshape(shape) % 257
            values_i = base_i + rank*1000
            require(invoke(attr(ring, 'fits_gather'), values_i, dim) is True, 'target-shaped gather does not fit ring')
            actual = cast(torch.Tensor, invoke(attr(comm, 'all_gather'), values_i, dim))
            compared += equal_bytes(actual, torch.cat([base_i, base_i+1000], dim=dim), 'dual-ring all-gather')
            count += 1
    torch.cuda.synchronize(device)
    return count, compared


def run(worker: object, *, rank_log: str, sysfs: Path = Path('/sys/class/infiniband')) -> Cases:
    r = runner(worker)
    distributed = importlib.import_module('vllm.distributed')
    rdma = importlib.import_module('vllm_exl3_moe.rdma')
    group = invoke(attr(distributed, 'get_tp_group'))
    comm = attr(group, 'device_communicator')
    rank = cast(int, attr(group, 'rank_in_group'))
    require(rank in (0, 1) and attr(group, 'world_size') == 2, 'real TP2 group required')
    require(invoke(attr(rdma, 'device_names')) == list(DEVICES), 'active ring devices differ')
    # ring_for could create a ring; inspect only the ring already established by warmup.
    rings = cast(dict[str, object], attr(rdma, '_rings'))
    ring = rings.get(cast(str, attr(comm, 'unique_name')))
    require(ring is not None, 'no established RDMA ring; refusing NCCL fallback')
    require(attr(attr(comm, 'all_reduce'), '_exl3_rdma') is True
        and attr(attr(comm, 'all_gather'), '_exl3_rdma') is True, 'actual communicator is not routed through RDMA')
    result: Cases = {f'gid-device{i}': gid(sysfs, name, 3) for i, name in enumerate(DEVICES)}
    result['rank-log'] = passed(1, line=log_line(rank_log, rank))
    before = counters(sysfs)
    config = attr(attr(worker, 'vllm_config'), 'model_config')
    hidden = cast(int, attr(attr(config, 'hf_config'), 'hidden_size'))
    samples, compared = collective(comm, ring, rank, hidden, cast(torch.device, attr(r, 'device')))
    dist = importlib.import_module('torch.distributed')
    invoke(attr(dist, 'barrier'), group=attr(group, 'cpu_group'))
    after = counters(sysfs)
    for i, name in enumerate(DEVICES):
        delta = after[name]-before[name]
        require(delta > 0, name+': no positive measured collective traffic, or counter reset')
        result[f'traffic-device{i}'] = passed(samples, bytes_delta=delta,
            before_bytes=before[name], after_bytes=after[name], unit='4-byte IB data counters converted to bytes')
    result['collective-bytes'] = passed(samples, byte_identical=True, bytes_compared=compared)
    return result
