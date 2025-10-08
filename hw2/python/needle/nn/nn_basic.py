"""The module.
"""
from typing import Any
from needle.autograd import Tensor, TensorOp
from needle import ops
import needle.init as init
import numpy as np
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True).transpose()) if bias else None

    def forward(self, X: Tensor) -> Tensor:
        return X.matmul(self.weight) + self.bias.broadcast_to(out.shape) if self.bias else out


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        return X.reshape((X.shape[0], math.prod(X.shape[1:])))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        return (lambda out: [out := m(out) for m in self.modules] and out)(x)

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        dim = len(logits.shape) - 1
        y_onehot = init.one_hot(logits.shape[dim], y, device=logits.device)
        log_softmax = ops.ops_logarithmic.logsoftmax(logits)
        ce = log_softmax * y_onehot
        loss = -ops.ops_mathematic.summation(ce, axes=(dim,))
        if dim == 1:
            size = logits.shape[0]
        else:
            size = math.prod(logits.shape[:dim])
        return ops.ops_mathematic.summation(loss) / size


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        mu = x.sum(axes=(0,)) / N
        xc = x - mu.broadcast_to(x.shape)
        var = (xc ** 2).sum(axes=(0,)) / N

        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.data
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var.data
            std = (var + self.eps) ** 0.5
            norm = xc / std.broadcast_to(x.shape)
        else:
            std = (self.running_var + self.eps) ** 0.5
            norm = (x - self.running_mean) / std

        g = self.weight.broadcast_to(x.shape)
        b = self.bias.broadcast_to(x.shape)
        return norm * g + b



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        size = math.prod(x.shape[1::])
        axes = tuple(range(1, len(x.shape)))
        e = ops.ops_mathematic.summation(x, axes=axes) / size
        dim = [x.shape[0]] + [1] * (len(x.shape) - 1)
        avg = ops.ops_mathematic.broadcast_to(e.reshape(dim), x.shape)
        std = (ops.ops_mathematic.summation((x - avg) ** 2, axes=axes) / size + self.eps) ** 0.5
        weight = ops.ops_mathematic.broadcast_to(self.weight, x.shape)
        bias = ops.ops_mathematic.broadcast_to(self.bias, x.shape)
        return (x - avg) / ops.ops_mathematic.broadcast_to(
            ops.ops_mathematic.reshape(std, dim), x.shape
        ) * weight + bias

class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x * init.randb(*x.shape, p = 1 - self.p) / (1 - self.p)
      

class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
