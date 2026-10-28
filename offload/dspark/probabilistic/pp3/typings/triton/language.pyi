from typing import Any, TypeAlias

constexpr: TypeAlias = Any  # noqa: PYI042 - public Triton API name
tensor: TypeAlias = Any  # noqa: PYI042 - public Triton API name
def __getattr__(name: str) -> Any: ...
