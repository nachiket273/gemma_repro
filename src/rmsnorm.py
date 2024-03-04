"""
Implementation of RMSNorm (https://arxiv.org/pdf/1910.07467.pdf)
"""
import torch
from torch.nn import Module, Parameter
from typing import AnyStr


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float=1e-6,
                 partial: float=-1, add_unit_offset: bool=True) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        if partial != -1 and (partial < 0 or partial > 1):
            raise ValueError("Value of parameter partial should be in range [0,1]")
        self.partial = partial
        self._init_parameter()

    def _init_parameter(self) -> None:
        self.scale =  Parameter(torch.ones(self.dim))
        self.register_parameter('scale', self.scale)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.type
        if self.partial != -1:
            pdim = int(self.dim * self.partial)
            x, _  = torch.split(x, [pdim, self.dim-pdim], dim=-1)
        x = x.float()
        norm = self._norm(x).type(dtype=dtype)
        if self.add_unit_offset:
            return norm * (1 + self.scale)
        return norm * self.scale

    def __repr__(self) -> AnyStr:
        return f"{self.__class__.__name__}(dim={self.dim}, eps={self.eps}, \
partial={self.partial}, add_unit_offset={self.add_unit_offset})"
