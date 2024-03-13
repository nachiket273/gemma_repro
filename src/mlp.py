"""
Multi-Layer Perceptron(MLP) Layer of transformer.
"""
from torch import Tensor
from torch.nn import Linear, Module
import torch.nn.functional as F


class MLP(Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.gelu(self.fc1(x), approximate='tanh')
        up = self.fc1(x)
        fused = gate * up
        out = self.fc2(fused)
        return out
