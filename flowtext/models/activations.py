import math

import oneflow as flow
from oneflow import nn


def gelu_new(x):
    return 0.5 * x * (1.0 + flow.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * flow.pow(x, 3.0))))


def gelu_fast(x):
    return 0.5 * x * (1.0 + flow.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def quick_gelu(x):
    return x * flow.sigmoid(1.702 * x)


def linear_act(x):
    return x


ACT2FN = {
    "relu": nn.functional.relu,
    "silu": nn.functional.silu,
    "swish": nn.functional.silu,
    "gelu": nn.functional.gelu,
    "tanh": flow.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "quick_gelu": quick_gelu,
    "mish": flow.mish,
    "linear": linear_act,
    "sigmoid": flow.sigmoid,
}
