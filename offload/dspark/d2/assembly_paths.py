"""Explicit relocation of immutable metadata; source documents and hashes stay unchanged."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import os


class AssemblyPaths:
    def __init__(self, root: Path, mappings: Mapping[str, str]) -> None:
        if root.is_symlink():
            raise ValueError('assembly root symlink refused')
        self.root = root.resolve(strict=True)
        self.mappings = dict(mappings)
        for old, relative in self.mappings.items():
            if not Path(old).is_absolute() or '..' in Path(old).parts:
                raise ValueError('relocation source must be an absolute normalized path')
            if Path(old) == Path('/') or self.root.is_relative_to(Path(old)):
                raise ValueError('relocation source prefix contains assembly root or is /')
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError('relocation destination must be relative to assembly root')
            self.inside(self.root / relative)

    def inside(self, path: Path) -> Path:
        path = Path(os.path.abspath(path))
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self.root):
            raise ValueError('identity path escapes assembly root: ' + str(path))
        # Symlinks are never identity aliases, including aliases within the root.
        if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(self.root)):
            raise ValueError('assembly identity symlink refused: ' + str(path))
        return resolved

    def resolve(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('identity path must be absolute and normalized')
        matches = [old for old in self.mappings if path.is_relative_to(Path(old))]
        if matches:
            old = max(matches, key=len)
            path = self.root / self.mappings[old] / path.relative_to(old)
        return self.inside(path)


def parse_mappings(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        old, sep, relative = value.partition('=')
        if not sep or not old or not relative or old in result:
            raise ValueError('path maps must be unique OLD_ABSOLUTE=ROOT_RELATIVE pairs')
        result[old] = relative
    return result
