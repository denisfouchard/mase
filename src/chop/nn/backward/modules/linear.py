import torch
from torch import Tensor
from .linear import QLinearFunction
from ..utils import clone_autograd_fn


class CustomLinear(torch.nn.Linear):
    """
    Linear module with custom autograd function
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(CustomLinear, self).__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.linear_autograd_fn = clone_autograd_fn(QLinearFunction)

    def forward(self, x: Tensor):
        return self.linear_autograd_fn.apply(x, self.weight, self.bias)
