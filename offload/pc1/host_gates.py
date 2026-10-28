"""Finite cat/inspect production probe and experiment setting gate; no port request.

Mounted launcher parsing follows reviewed UP3 06be7c0. Docker Cmd is only
['/serve.sh']; it is not the vLLM argv. Captured UP3 fixtures retain provenance.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from typing import cast

from offload.launch.windows import up2_checks as ck
from offload.pc1 import build as b
from offload.prefill_maxlen.evidence import Json, record

admit = b.admit


def serve_arguments(script: str) -> list[str]:
    import re
    commands = [shlex.split(line, comments=True) for line in re.sub(r'\\\r?\n', ' ', script).splitlines()]
    launches = [tokens for tokens in commands if tokens[:3] == ['exec', 'vllm', 'serve']]
    ck.require(len(launches) == 1, 'one exec vllm serve required in mounted /serve.sh')
    ck.require(sum(t == '--port' or t.startswith('--port=') for command in commands for t in command) == 1,
               'ambiguous port across mounted launcher commands')
    return launches[0]


def settings(script: str, rank: int, port: str, arm: str) -> dict[str, Json]:
    admit(arm)
    argv = serve_arguments(script)
    observed: dict[str, Json] = {}
    for key, value in zip(('--port', '--node-rank', '--max-num-batched-tokens', '--long-prefill-token-threshold'),
                          (port, str(rank), *map(str, b.SETTINGS[arm]))):
        ck.require(argv.count(key) == 1 and not any(t.startswith(key + '=') for t in argv),
                   'missing/ambiguous literal ' + key)
        index = argv.index(key) + 1
        ck.require(index < len(argv) and argv[index] == value, 'mounted launcher setting differs: ' + key)
        observed[key.removeprefix('--')] = value
    return observed


class ProductionReads:
    def __init__(self, host: str | None = None) -> None:
        ck.require(host in (None, 'sparkhead', 'sparkworker-wired'), 'registered Spark required')
        self.host = host

    def output(self, argv: list[str]) -> str:
        ck.require(argv in (['docker', 'inspect', 'dsv41-serve'],
            ['docker', 'exec', 'dsv41-serve', 'cat', '/serve.sh']), 'probe command outside finite allowlist')
        command = argv if self.host is None else [
            'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', self.host, shlex.join(argv)]
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, errors='replace', timeout=15)


def inspect_running(text: str, container: str | None = None) -> dict[str, Json]:
    rows = ck.rows(cast(Json, json.loads(text)))
    ck.require(len(rows) == 1, 'exactly one inspected container required')
    info = rows[0]
    ck.require(record(info['State']).get('Running') is True and isinstance(info.get('Id'), str)
        and bool(info['Id']) and (container is None or info['Id'] == container), 'container changed/not running')
    config = record(info['Config'])
    ck.require(config.get('Cmd') == ['/serve.sh'] and config.get('Entrypoint') == ['/bin/bash'],
               'Docker wrapper differs from captured production shape')
    env: dict[str, str] = {}
    for value in ck.strings(config['Env']):
        key, equal, item = value.partition('=')
        ck.require(bool(key) and equal == '=' and key not in env, 'malformed/duplicate Docker environment')
        env[key] = item
    return info


def collect(arm: str, rank: int, container: str | None = None, *, reads: ProductionReads | None = None,
            production: bool = False) -> dict[str, Json]:
    admit(arm)
    ck.require(rank in (0, 1), 'rank0 or1 required')
    transport = reads or ProductionReads()
    ck.require(transport.host in (None, ('sparkhead', 'sparkworker-wired')[rank]), 'host/rank mismatch')
    ck.require(not production or arm == 'control' and container is None, 'production/experiment modes cannot mix')
    info = inspect_running(transport.output(['docker', 'inspect', 'dsv41-serve']), container)
    script = transport.output(['docker', 'exec', 'dsv41-serve', 'cat', '/serve.sh'])
    observed = settings(script, rank, '8000' if production else '8001', arm)
    inspect_running(transport.output(['docker', 'inspect', 'dsv41-serve']), str(info['Id']))
    return {'mode': 'probe-production' if production else 'experiment-settings', 'status': 'PARSE COMPLETE',
        'qualification': 'NOT QUALIFIED: plumbing only; no correctness/performance/restore evidence',
        'arm': arm, 'rank': rank, 'container': info['Id'], 'serve': observed, 'identity_rechecked': True}


def validate(receipt: dict[str, Json], arm: str, rank: int, container: str) -> None:
    ck.require(receipt.get('mode') == 'experiment-settings' and receipt.get('status') == 'PARSE COMPLETE'
        and receipt.get('arm') == arm and receipt.get('rank') == rank and receipt.get('container') == container
        and receipt.get('identity_rechecked') is True, 'invalid experiment setting evidence')
    ck.require(record(receipt['serve']) == dict(zip(
        ('port', 'node-rank', 'max-num-batched-tokens', 'long-prefill-token-threshold'),
        ('8001', str(rank), *map(str, b.SETTINGS[arm])))), 'experiment settings differ')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-production', action='store_true')
    parser.add_argument('--host', choices=('sparkhead', 'sparkworker-wired'))
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--arm', choices=b.ARMS)
    parser.add_argument('--container')
    args = parser.parse_args(argv)
    if args.probe_production:
        if args.arm is not None or args.container is not None:
            parser.error('probe and experiment modes cannot mix')
        result = collect('control', args.rank, reads=ProductionReads(args.host), production=True)
    else:
        if args.host is not None or args.arm is None or not args.container:
            parser.error('experiment requires --arm/--container locally')
        result = collect(args.arm, args.rank, args.container)
        validate(result, args.arm, args.rank, args.container)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
