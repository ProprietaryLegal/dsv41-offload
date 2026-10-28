"""CPU tests for the logical-bytes model, the ncu ratio report and the KL1-W rate re-derivation (synthetic files)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from offload.census import bytes_model as BM
from offload.kl1 import shapes as S
from offload.prefill_maxlen.evidence import Json, number, record

GU = '(anonymous namespace)::smoe_gu((anonymous namespace)::Layer, (anonymous namespace)::Call)'
DOWN = '(anonymous namespace)::smoe_down((anonymous namespace)::Layer, (anonymous namespace)::Call)'
MROW5 = 'void (anonymous namespace)::mrow_kernel<5, false>(__nv_bfloat16 const*, int)'
HEAD = 'void (anonymous namespace)::mrow_kernel<6, false>(__nv_bfloat16 const*, int)'


def test_floor_reproduces_plan_section_2_2() -> None:
    plan = BM.floor(1)
    mb = record(plan['mb_per_layer'])
    assert number(mb['routed_6_experts']) == pytest.approx(36.7, abs=0.05)
    assert number(mb['shared']) == pytest.approx(11.1, abs=0.05)
    assert number(mb['dense']) == pytest.approx(42.5, abs=0.05)
    assert number(mb['router']) == pytest.approx(3.9, abs=0.05)
    assert number(mb['mhc']) == pytest.approx(2.0, abs=0.05)
    assert number(plan['lm_head_mb']) == pytest.approx(248, abs=0.5)
    assert number(plan['total_gb']) == pytest.approx(4.10, abs=0.005)
    floors = record(plan['floor_ms'])
    assert [round(number(floors[k]), 1) for k in ('kl1_isolated_226.6', 'source_230', 'spec_273')] == [18.1, 17.8, 15.0]
    share = record(plan['non_streaming_share'])
    assert round(number(share['kl1_isolated_226.6@24.3ms']), 2) == 0.26
    assert round(number(share['kl1_isolated_226.6@27.7ms']), 2) == 0.35
    assert round(number(share['spec_273@24.3ms']), 2) == 0.38
    assert round(number(share['spec_273@27.7ms']), 2) == 0.46
    two = BM.floor(2)
    assert number(two['total_gb']) - number(plan['total_gb']) == pytest.approx(40 * S.MHC_BYTES / 1e9)


def test_step_and_round_bytes() -> None:
    b = BM.step_bytes(13, 6)
    assert b['total'] == pytest.approx(sum(v for k, v in b.items() if k != 'total'))
    assert b['shared'] == pytest.approx(40 * S.expert_bytes(S.SHARED_K))           # one run of <= 8 rows
    assert BM.step_bytes(13, 9)['shared'] == pytest.approx(2 * b['shared'])         # two runs
    r = BM.round_bytes(6, 13)
    assert number(record(r['uniform'])['distinct']) == pytest.approx(34.6, abs=0.05)
    assert number(record(r['given'])['floor_ms_at_226.6']) == pytest.approx(b['total'] / 226.6e6)


def test_logical_specs() -> None:
    gu, dn = BM.expert_mean()
    sgu, sdn = BM.shared_parts()
    assert BM.logical({'model': 'smoe_gu', 'rows': 6, 'distinct': 12}, 0) == pytest.approx(12 * gu + sgu)
    assert BM.logical({'model': 'smoe_down', 'rows': 9, 'distinct': 12, 'shared': False}, 0) == pytest.approx(12 * dn)
    assert BM.logical({'model': 'smoe_layer', 'rows': 6, 'distinct': 12, 'k': 2}, 0) == pytest.approx(
        12 * S.expert_bytes(2) + S.expert_bytes(S.SHARED_K))
    assert [BM.logical({'model': 'dense_cycle'}, i) for i in range(5)] == [
        lin.bytes for lin in (*S.DECODE_DENSE, S.DECODE_DENSE[0])]
    assert BM.logical({'model': 'lm_head'}, 0) == S.LM_HEAD.bytes
    assert BM.logical({'bytes': 123.0}, 7) == 123.0
    with pytest.raises(ValueError, match='unknown'):
        BM.logical({'model': 'nope'}, 0)


RAW = '\n'.join([
    '==PROF== Connected to process 1',
    '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device",'
    '"CC","dram__bytes_read.sum","dram__bytes_write.sum","gpu__time_duration.sum","launch__func_cache_config"',
    '"","","","","","","","","","","","Mbyte","Kbyte","usecond",""',
    f'"0","1","python","h","{GU}","1","7","(288, 1, 1)","(144, 1, 1)","0","12.1","{{gu}}","10","200","CachePreferNone"',
    f'"1","1","python","h","{DOWN}","1","7","(288, 1, 1)","(40, 1, 1)","0","12.1","{{dn}}","10","100","CachePreferNone"',
    f'"2","1","python","h","{HEAD}","1","7","(128, 1, 1)","(48, 1, 1)","0","12.1","{{head}}","0","1000","x"',
    '"3","1","python","h","unrelated_kernel","1","7","(1, 1, 1)","(1, 1, 1)","0","12.1","1","0","1","x"',
])


def details(gu_bytes: float) -> str:
    head = ('"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size",'
            '"Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"')
    rows = [f'"0","1","p","h","{GU}","1","7","(288, 1, 1)","(144, 1, 1)","0","12.1","Command line profiler metrics",'
            f'"lts__t_sectors_srcunit_tex_op_read.sum","sector","{gu_bytes / 32:,.0f}"',
            f'"0","1","p","h","{GU}","1","7","(288, 1, 1)","(144, 1, 1)","0","12.1","Command line profiler metrics",'
            '"gpu__time_duration.sum","nsecond","200,000"']
    return '\n'.join([head, *rows])


def test_read_ncu_csv_raw_and_details() -> None:
    raw = BM.read_ncu_csv(RAW.replace('{gu}', '50').replace('{dn}', '25').replace('{head}', '248'))
    assert [x.family for x in raw] == ['smoe_gu', 'smoe_down', 'mrow_head', 'rest']
    assert raw[0].metrics['dram__bytes_read.sum'] == pytest.approx(50e6)
    assert raw[0].metrics['dram__bytes_write.sum'] == pytest.approx(10e3)
    assert raw[0].metrics['gpu__time_duration.sum'] == pytest.approx(200e-6)
    assert raw[0].grid == (144, 1, 1)
    det = BM.read_ncu_csv(details(3.2e6))
    assert len(det) == 1 and det[0].metrics['lts__t_sectors_srcunit_tex_op_read.sum'] == pytest.approx(3.2e6)
    with pytest.raises(ValueError, match='lacks column'):
        BM.read_ncu_csv('"A","B"\n"1","2"')


def _manifest(tmp_path: Path, text: str, extra: dict[str, Json] | None = None, sha: bool = True) -> Path:
    csv_path = tmp_path / 'run.csv'
    csv_path.write_text(text)
    run: dict[str, Json] = {'csv': 'run.csv', 'label': 'smoe r6 u12',
                            'launches': [{'match': 'smoe_gu', 'logical': {'model': 'smoe_gu', 'rows': 6,
                                                                          'distinct': 12}},
                                         {'match': 'smoe_down', 'logical': {'model': 'smoe_down', 'rows': 6,
                                                                            'distinct': 12}},
                                         {'match': 'mrow_head', 'logical': {'model': 'lm_head'}}]}
    if sha:
        run['sha256'] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    m: dict[str, Json] = {'type': 'cen3-ncu-manifest', 'clock_cap_mhz': 2300, 'host': 'sparkhead', 'runs': [run],
                          **(extra or {})}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(m))
    return path


def test_ratio_report_verdicts(tmp_path: Path) -> None:
    gu = BM.logical({'model': 'smoe_gu', 'rows': 6, 'distinct': 12}, 0)
    dn = BM.logical({'model': 'smoe_down', 'rows': 6, 'distinct': 12}, 0)
    text = (RAW.replace('{gu}', f'{gu * 1.05 / 1e6:.6f}').replace('{dn}', f'{dn * 1.30 / 1e6:.6f}')
            .replace('{head}', f'{S.LM_HEAD.bytes * 0.5 / 1e6:.6f}'))
    res = BM.ratio_report(_manifest(tmp_path, text))
    fam = record(res['families'])
    assert str(record(fam['smoe_gu'])['verdict']).startswith('PHYSICAL ~ LOGICAL')
    assert str(record(fam['smoe_down'])['verdict']).startswith('OVER-FETCH')
    assert str(record(fam['mrow_head'])['verdict']).startswith('BELOW LOGICAL')
    assert number(record(fam['smoe_down'])['read_over_logical']) == pytest.approx(1.30, abs=1e-4)
    runs = res['runs']
    assert isinstance(runs, list)
    run = record(runs[0])
    assert run['unassigned_launches'] == 1
    gu_row = record(record(run['families'])['smoe_gu'])
    assert number(gu_row['logical_gbps']) == pytest.approx(gu / 200e-6 / 1e9)
    assert res['physical_source'] == 'DRAM counters'


def test_ratio_report_fallback_and_refusals(tmp_path: Path) -> None:
    gu = BM.logical({'model': 'smoe_gu', 'rows': 6, 'distinct': 12}, 0)
    path = _manifest(tmp_path, details(gu), {'read_metrics': ['lts__t_sectors_srcunit_tex_op_read.sum']})
    res = BM.ratio_report(path)
    assert str(res['physical_source']).startswith('L2 fallback')
    assert number(record(record(res['families'])['smoe_gu'])['read_over_logical']) == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(ValueError, match='query-metrics'):
        BM.ratio_report(_manifest(tmp_path, details(gu)))                                # DRAM metric absent
    path = _manifest(tmp_path, details(gu), {'read_metrics': ['lts__t_sectors_srcunit_tex_op_read.sum']})
    (tmp_path / 'run.csv').write_text(details(gu * 2))
    with pytest.raises(ValueError, match='sha256'):
        BM.ratio_report(path)


def test_kl1_window_rates(tmp_path: Path) -> None:
    root = tmp_path / 'KL1-W-20990101T000000Z'
    host = root / 'results' / 'sparkhead'
    host.mkdir(parents=True)
    smoe = {'type': 'kl1-smoe-bench', 'variant': 'prod',
            'results': [{'case': 'r6q1u12c0', 'rows': 6, 'requests': 1, 'distinct': 12, 'bytes_per_layer': 90e6,
                         'us_per_layer_median': 400.0}]}
    dense = {'type': 'kl1-dense-bench', 'rows': [{'variant': 'ring', 'min32': 9, 'rows': 4, 'us_per_layer': 200.0,
                                                 'gbps': 1.0}]}
    files = {'prod-first.json': smoe, 'dense.json': dense, 'failures.json': []}
    hashes: dict[str, Json] = {}
    for name, doc in files.items():
        (host / name).write_text(json.dumps(doc))
        hashes[f'results/sparkhead/{name}'] = hashlib.sha256((host / name).read_bytes()).hexdigest()
    (root / 'summary.json').write_text(json.dumps({'artifact_sha256': hashes}))
    res = BM.kl1_window(root)
    h = record(record(res['hosts'])['sparkhead'])
    assert number(record(h['prod-first'])['logical_gbps_median']) == pytest.approx(90e6 / 400 / 1e3)
    d = record(h['dense'])
    assert number(d['layer_mb']) == pytest.approx(42.5, abs=0.05)
    assert record(h['failures'])['failures'] == 0
    (host / 'dense.json').write_text('{}')
    with pytest.raises(ValueError, match='sha256'):
        BM.kl1_window(root)


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert BM.main(['floor']) == 0
    assert 'plan_one_mhc_weight' in capsys.readouterr().out
    assert BM.main(['step', '--rows', '6', '--distinct', '13']) == 0
    out = tmp_path / 'x.json'
    out.write_text('{}')
    assert BM.main(['ratio', '--manifest', str(tmp_path / 'm.json'), '--output', str(out)]) == 1
