import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    return rand(fan_in, fan_out, low=-gain * math.sqrt(6 / (fan_in + fan_out)), high=gain * math.sqrt(6 / (fan_in + fan_out)), **kwargs)


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    return randn(fan_in, fan_out, mean=0, std=gain * math.sqrt(2 / (fan_in + fan_out)), **kwargs)

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    return rand(fan_in, fan_out, low=-math.sqrt(6 / fan_in), high=math.sqrt(6 / fan_in), **kwargs)



def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    return randn(fan_in, fan_out, mean=0, std=math.sqrt(2 / fan_in), **kwargs)