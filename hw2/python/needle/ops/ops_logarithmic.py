from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        shape = list(Z.shape)
        shape[1] = 1
        self.lse = logsumexp(Tensor(Z), axes=(1,))
        return Z - numpy.reshape(self.lse.numpy(), tuple(shape))

    def gradient(self, out_grad: Tensor, node: Tensor):
        lse_grad = self.lse.op.gradient(summation(out_grad, axes=(1,)), self.lse)
        return out_grad - lse_grad


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        M = array_api.max(Z, axis=self.axes, keepdims=True)
        Z_ = Z - M
        E = array_api.exp(Z_)
        self.exp = Tensor(E)
        S = array_api.sum(E, axis=self.axes, keepdims=True)
        self.sum = Tensor(S)
        return array_api.log(array_api.squeeze(S, axis=self.axes)) + array_api.squeeze(
            M, axis=self.axes
        )

    def gradient(self, out_grad: Tensor, node: Tensor):
        s = list(node.inputs[0].shape)
        s = [1 if i in self.axes else s[i] for i in range(len(s))] if self.axes else [1] * len(s)
        G = broadcast_to(reshape(out_grad, s), self.exp.shape)
        return G * self.exp / self.sum


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)