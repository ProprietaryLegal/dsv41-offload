"""Independent CPU E2M1/E8M0 decoder and admission; no CUDA import or access."""
from __future__ import annotations

import torch

E2M1: tuple[float, ...] = (0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.)


def decode_codes(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Float64 oracle, including signed zero, scale-0 and NaN scale-255."""
    if codes.device.type != 'cpu' or scales.device.type != 'cpu':
        raise ValueError('reference decoder requires CPU tensors')
    values = torch.tensor(E2M1, dtype=torch.float64)[codes.long()]
    values = torch.ldexp(values, scales.to(torch.int32) - 127)
    return torch.where(scales == 255, float('nan'), values)


def unpack(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError('checkpoint codes and E8M0 scales must be uint8')
    if packed.ndim != 3 or scales.shape != (*packed.shape[:2], packed.shape[2] // 16):
        raise ValueError('expected [E,N,K/2] codes and [E,N,K/32] scales')
    if packed.shape[-1] % 16:
        raise ValueError('K must be a multiple of group size 32')
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    return decode_codes(codes, scales.repeat_interleave(32, dim=-1))


def admit(packed: torch.Tensor, scales: torch.Tensor) -> dict[str, int]:
    """Bounded load-time validation. Never narrow/repair unsupported native values.

    Inspect one expert at a time so preparing a full bank never expands the
    checkpoint in RAM. BF16 subnormals decode exactly but MMA may flush them:
    reject them for computation rather than asserting unsupported fidelity.
    """
    if packed.device.type != 'cpu' or scales.device.type != 'cpu':
        raise ValueError('admission must run on the CPU before GPU allocation')
    if packed.ndim != 3 or packed.shape[0] == 0 or packed.shape[1] == 0:
        raise ValueError('expected nonempty expert bank')
    result = {'values': 0, 'negative_zeros': 0, 'scale_min': 255, 'scale_max': 0}
    for e in range(packed.shape[0]):
        decoded = unpack(packed[e:e+1], scales[e:e+1])
        if not bool(torch.isfinite(decoded).all()):
            raise ValueError(f'expert {e}: NaN/nonfinite native scale/value')
        widened = decoded.to(torch.bfloat16).to(torch.float64)
        if not torch.equal(decoded, widened):
            raise ValueError(f'expert {e}: native values not exactly representable in BF16')
        subnormal = (decoded != 0) & (decoded.abs() < torch.finfo(torch.bfloat16).tiny)
        if bool(subnormal.any()):
            raise ValueError(f'expert {e}: BF16 subnormal operands require a wider arithmetic path')
        result['values'] += decoded.numel()
        result['negative_zeros'] += int(((decoded == 0) & decoded.signbit()).sum())
        result['scale_min'] = min(result['scale_min'], int(scales[e].min()))
        result['scale_max'] = max(result['scale_max'], int(scales[e].max()))
    return result
