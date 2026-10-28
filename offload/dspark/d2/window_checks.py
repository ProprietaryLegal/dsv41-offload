"""CPU checks closing replay check's missing-file and partial-equivalence gaps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from collections.abc import Mapping


def response_policy(response: Mapping[str, object]) -> None:
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError('refused or malformed replay response: expected one choice')
    choice = choices[0]
    if response.get('error') or response.get('refusal') or choice.get('refusal'):
        raise ValueError('refused replay response')
    # None preserves old fake/legacy response fixtures; explicit error/cancel/filter finishes refuse.
    if choice.get('finish_reason') not in (None, 'stop', 'length'):
        raise ValueError('refused replay finish reason')
    usage = response.get('usage') or {}
    if not isinstance(usage, dict):
        raise ValueError('malformed replay usage')
    details = usage.get('prompt_tokens_details') or {}
    if not isinstance(details, dict):
        raise ValueError('malformed replay cache accounting')
    if details.get('cached_tokens'):
        raise RuntimeError('cached prompt tokens: replay rows were not recomputed')


def response_entry(meta: Mapping[str, object], entry: Mapping[str, object]) -> None:
    for key in ('seq_id', 'request_id', 'L', 'gen_start', 'from', 'tokens_sha256'):
        if meta[key] != entry[key]:
            raise ValueError('response metadata differs from replay plan: '+key)
    usage = meta.get('usage', {})
    if not isinstance(usage, dict):
        raise ValueError('missing response usage')
    if (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) != 0:
        raise ValueError('cached response refused')


def shard_complete(text: str, records: int, k: int) -> None:
    start = re.search(r'^(\d+) blocks in shard '+str(k)+r'/3, (\d+) to go$', text, re.M)
    end = re.fullmatch(r'done (\d+) blocks in [0-9]+ s', text.splitlines()[-1] if text else '')
    if not start or not end or start[1] != start[2] or end[1] != start[2]:
        raise ValueError('equivalence shard did not finish a fresh complete population')
    if records != int(start[1]):
        raise ValueError('partial equivalence shard records')


def responses(root: Path) -> None:
    plan = [json.loads(line) for line in (root/'plan.jsonl').read_text().splitlines()]
    if not plan:
        raise ValueError('empty plan')
    for entry in plan:
        meta = json.loads((root/'logprobs'/(entry['seq_id']+'.json')).read_text())
        response_entry(meta, entry)
    print(json.dumps({'responses_checked': len(plan), 'cached': 0}))


def equiv_complete(work: Path) -> None:
    total = 0
    for k in range(3):
        text = (work/f'drafts-{k}.log').read_text()
        records = (work/f'drafts-{k}of3.jsonl').read_text().splitlines()
        shard_complete(text, len(records), k)
        total += len(records)
    if total != 542:
        raise ValueError(f'expected all 542 A-capture blocks, found {total}')
    print(json.dumps({'complete_shards': 3, 'blocks': total, 'every': 1}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('check', choices=('responses', 'equiv-complete'))
    ap.add_argument('root', type=Path)
    a = ap.parse_args()
    (responses if a.check == 'responses' else equiv_complete)(a.root)


if __name__ == '__main__':
    main()
