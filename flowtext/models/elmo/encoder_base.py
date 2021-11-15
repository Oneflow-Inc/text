import oneflow as flow
from oneflow.nn import Module
from .utils import get_lengths_from_binary_sequence_mask, sort_batch_by_length
import torch
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from typing import Tuple, Optional, Callable, Union


RnnState = Union[flow.Tensor, Tuple[flow.Tensor, flow.Tensor]]
RnnStateStorage = Tuple[flow.Tensor, ...]


class _EncoderBase(Module):
    def __init__(self, stateful: bool = False):
        super().__init__()
        self.stateful = stateful
        self._states: Optional[RnnStateStorage] = None

    # TODO: use pytorch instead, modify after oneflow support.
    def sort_and_run_forward(
        self,
        module: Callable[
            [PackedSequence, Optional[RnnState]],
            Tuple[Union[PackedSequence, flow.Tensor], RnnState],
        ],
        inputs: flow.Tensor,
        mask: flow.Tensor,
        hidden_state: Optional[RnnState] = None,
    ):
        batch_size = mask.size(0)
        num_valid = flow.sum(mask[:, 0]).int().item()
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        (
            sorted_inputs,
            sorted_sequence_lengths,
            restoration_indices,
            sorting_indices,
        ) = sort_batch_by_length(inputs, sequence_lengths)
        # TODO: use pytorch instead, modify after oneflow support.
        packed_sequence_input = pack_padded_sequence(
            torch.tensor(sorted_inputs[:num_valid, :, :].numpy()),
            sorted_sequence_lengths[:num_valid].data.tolist(),
            batch_first=True,
        )

        if not self.stateful:
            if hidden_state is None:
                initial_states = hidden_state
            elif isinstance(hidden_state, tuple):
                initial_states = [
                    state.index_select(1, sorting_indices)[:, :num_valid, :]
                    for state in hidden_state
                ]
            else:
                initial_states = hidden_state.index_select(1, sorting_indices)[
                    :, :num_valid, :
                ]
        else:
            initial_states = self._get_initial_states(
                batch_size, num_valid, sorting_indices
            )

        module_output, final_states = module(packed_sequence_input, initial_states)

        return module_output, final_states, restoration_indices

    def _get_initial_states(
        self, batch_size: int, num_valid: int, sorting_indices: flow.Tensor
    ) -> Optional[RnnState]:
        if self._states is None:
            return None
        if batch_size > self._states[0].size(1):
            num_states_to_concat = batch_size - self._states[0].size(1)
            resized_states = []
            for state in self._states:
                zeros = flow.zeros(
                    state.size(0),
                    num_states_to_concat,
                    state.size(2),
                    dtype=state.dtype,
                ).to(state.device)
                zeros = flow.Tensor(zeros)
                resized_states.append(flow.cat([state, zeros], 1))
            self._states = tuple(resized_states)
            correctly_shaped_states = self._states
        elif batch_size < self._states[0].size(1):
            correctly_shaped_states = tuple(
                state[:, :batch_size, :] for state in self._states
            )
        else:
            correctly_shaped_states = self._states
        if self._states.size(0) == 1:
            correctly_shaped_state = correctly_shaped_states[0]
            sorted_state = correctly_shaped_state.index_select(1, sorting_indices)
            return sorted_state[:, :num_valid, :]
        else:
            sorted_states = [
                state.index_select(1, sorting_indices)
                for state in correctly_shaped_states
            ]
            return tuple(state[:, :num_valid, :] for state in sorted_states)

    def _update_states(
        self, final_states: RnnStateStorage, restoration_indices: flow.Tensor
    ) -> None:
        new_unsorted_states = [
            state.index_select(1, restoration_indices) for state in final_states
        ]
        if self._states is None:
            self._states = tuple(
                [
                    state.clone().detach().requires_grad_(True)
                    for state in new_unsorted_states
                ]
            )
        else:
            current_state_batch_size = self._states[0].size(1)
            new_state_batch_size = final_states[0].size(1)
            used_new_rows_mask = [
                (state[0, :, :].sum(-1) != 0.0).float().view(1, new_state_batch_size, 1)
                for state in new_unsorted_states
            ]
            new_states = []
            if current_state_batch_size > new_state_batch_size:
                for old_state, new_state, used_mask in zip(
                    self._states, new_unsorted_states, used_new_rows_mask
                ):
                    masked_old_state = old_state[:, :new_state_batch_size, :] * (
                        1 - used_mask
                    )
                    old_state[:, :new_state_batch_size, :] = (
                        new_state + masked_old_state
                    )
                    new_states.append(flow.Tensor(old_state.data))
            else:
                new_states = []
                for old_state, new_state, used_mask in zip(
                    self._states, new_unsorted_states, used_new_rows_mask
                ):
                    masked_old_state = old_state * (1 - used_mask)
                    new_state += masked_old_state
                    new_states.append(flow.Tensor(new_state.data))
            self._states = tuple(new_states)

    def reset_states(self):
        self._states = None
