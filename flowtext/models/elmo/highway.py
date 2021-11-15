import oneflow as flow
from typing import Callable
from overrides import overrides


class Highway(flow.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 1,
        activation: Callable[[flow.Tensor], flow.Tensor] = flow.nn.functional.relu,
    ) -> None:
        super(Highway, self).__init__()
        self._input_dim = input_dim
        self._layers = flow.nn.ModuleList(
            [flow.nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self._activation = activation
        for layer in self._layers:
            layer.bias[input_dim:].data.fill_(1)

    @overrides
    def forward(self, inputs):
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part = projected_input[
                :, (0 * self._input_dim) : (1 * self._input_dim)
            ]
            gate = projected_input[:, (1 * self._input_dim) : (2 * self._input_dim)]
            nonlinear_part = self._activation(nonlinear_part)
            gate = flow.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input
