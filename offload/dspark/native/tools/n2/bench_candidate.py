"""Explicit measured-small experiment; production/default arm is not mutated on disk."""
from __future__ import annotations

from dataclasses import replace
import sys

from ...tp2_native import ARMS
from ..native_bench import main


if __name__ == '__main__':
    output, rank = sys.argv[1:3]
    ARMS['native-hybrid-kx2'] = replace(ARMS['native-hybrid-kx2'], dense_profile='measured-small')
    sys.argv = [sys.argv[0], 'native-hybrid-kx2', 'solo', output, rank, '200']
    main()
