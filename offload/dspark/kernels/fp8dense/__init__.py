"""Source-faithful FP8 block-scaled dense GEMM on SM86. See README.md."""
from .api import DenseFP8, extension

__all__ = ['DenseFP8', 'extension']
