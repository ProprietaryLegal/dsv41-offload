from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from offload.prefill_maxlen.evidence import record
from offload.up3 import engine_ring as r


def fabric(root: Path, device: str, ipv4: str) -> None:
    port = root/device/'ports/1'
    for path, value in {f'gids/3': '::ffff:'+ipv4, 'gid_attrs/types/3': 'RoCE v2',
                        'gid_attrs/ndevs/3': 'eth-fixture', 'counters/port_xmit_data': '11',
                        'counters/port_rcv_data': '17'}.items():
        dest = port/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(value+'\n')


def test_dual_gid_log_and_real_counter_units(tmp_path: Path) -> None:
    for device, address in zip(r.DEVICES, ('10.202.26.2', '10.202.27.2')):
        fabric(tmp_path, device, address)
        assert record(r.gid(tmp_path, device, 3))['ipv4_roce_v2'] is True
    assert r.counters(tmp_path) == {d: (11+17)*4 for d in r.DEVICES}
    line = 'EXL3 RDMA all-reduce: rank 1 of 2 on '+','.join(r.DEVICES)+', gid 3, proxy on cpu 14, up to 256 KB'
    assert r.log_line('(worker prefix) '+line, 1) == line
    for bad in (line.replace('gid 3', 'gid 1'), line+'\n'+line, line+'\nRDMA all-reduce off NCCL stays'):
        with pytest.raises(ValueError):
            r.log_line(bad, 1)
    with pytest.raises(ValueError):
        r.log_line(line, 0)
    fabric(tmp_path, r.DEVICES[1], '10.202.29.2')
    with pytest.raises(ValueError, match='fabric'):
        r.gid(tmp_path, r.DEVICES[1], 3)


@pytest.mark.parametrize('rank', [0, 1])
def test_collective_every_byte_all_rows_and_dimensions(rank: int) -> None:
    def reduce(x: torch.Tensor) -> torch.Tensor:
        return (x-rank)*2+1
    def gather(x: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.cat([x-rank*1000, x+(1-rank)*1000], dim=dim)
    comm = SimpleNamespace(all_reduce=reduce, all_gather=gather)
    ring = SimpleNamespace(rank=rank, world=2, fits=lambda x: True, fits_gather=lambda x, dim: True)
    with patch.object(torch.cuda, 'synchronize'):
        count, compared = r.collective(comm, ring, rank, 5120, torch.device('cpu'))
    assert count == 7 and compared > 24*5120*2
    def broken(x: torch.Tensor) -> torch.Tensor:
        y = reduce(x)
        y[-1] += 1
        return y
    comm.all_reduce = broken
    with pytest.raises(ValueError, match='bytes differ'):
        r.collective(comm, ring, rank, 5120, torch.device('cpu'))


def test_missing_gid_and_non_v2_fail(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        r.gid(tmp_path, r.DEVICES[0], 3)
    fabric(tmp_path, r.DEVICES[0], '10.202.26.2')
    (tmp_path/r.DEVICES[0]/'ports/1/gid_attrs/types/3').write_text('RoCE v1')
    with pytest.raises(ValueError, match='RoCE v2'):
        r.gid(tmp_path, r.DEVICES[0], 3)
