from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get('TV1_GPU_TEST') != '1', reason='locked x86host toolchain only')
def test_offline_sm121_compile() -> None:
    import triton  # type: ignore[import-untyped]
    from triton.backends.compiler import GPUTarget  # type: ignore[import-untyped]
    from triton.compiler.compiler import ASTSource  # type: ignore[import-untyped]
    from offload.tv1.candidate_kernel import _score
    constants = {'QS': 4096, 'QH': 128, 'KS': 8704, 'WS': 32, 'PS': 2350, 'CS': 2048,
                 'OS': 150000, 'PAGE': 64, 'NC': 2048, 'WIDTH': 150000,
                 'HEADS': 32, 'BLOCK': 8, 'TILE': 32}
    source = ASTSource(fn=_score, signature={'Q': '*u8', 'K': '*u8', 'W': '*fp32', 'L': '*i32',
                                           'P': '*i32', 'C': '*i32', 'O': '*fp32'}, constexprs=constants)
    compiled = triton.compile(source, target=GPUTarget('cuda', 121, 32),
                              options={'num_warps': 4, 'enable_fp_fusion': False})
    assert '.target sm_121' in compiled.asm['ptx']
    assert compiled.metadata.shared <= 48 * 1024
    # Never access compiled.run/n_regs: that would try loading SM121 code on SM86.
    destination = os.environ.get('TV1_COMPILE_RECEIPT')
    if destination:
        Path(destination).write_text(json.dumps({'target': 'sm121', 'shared_bytes': compiled.metadata.shared,
            'ptx_bytes': len(compiled.asm['ptx']), 'cubin_bytes': len(compiled.asm['cubin']),
            'executed_on_sm121': False}) + '\n')
