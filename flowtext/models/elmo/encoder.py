import oneflow as flow
from .encoder_base import _EncoderBase
from .lstm_cell import LstmCellWithProjection
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from typing import Optional, Tuple


class ElmobiLm(_EncoderBase):
    def __init__(self, config, use_cuda=False):
        super().__init__(stateful=True)
        self.config = config
        self.use_cuda = use_cuda
        input_size = config["encoder"]["projection_dim"]
        hidden_size = config["encoder"]["projection_dim"]
        cell_size = config["encoder"]["dim"]
        num_layers = config["encoder"]["n_layers"]
        memory_cell_clip_value = config['encoder']['cell_clip']
        state_projection_clip_value = config['encoder']['proj_clip']
        recurrent_dropout_probability = config['dropout']
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_size = cell_size
        
        forward_layers = []
        backward_layers = []
        
        lstm_input_size = input_size
        go_forward = True
        for layer_index in range(num_layers):
            forward_layer = LstmCellWithProjection(lstm_input_size,
                                                   hidden_size,
                                                   cell_size,
                                                   go_forward,
                                                   recurrent_dropout_probability,
                                                   memory_cell_clip_value,
                                                   state_projection_clip_value)
            backward_layer = LstmCellWithProjection(lstm_input_size,
                                                    hidden_size,
                                                    cell_size,
                                                    not go_forward,
                                                    recurrent_dropout_probability,
                                                    memory_cell_clip_value,
                                                    state_projection_clip_value)
            lstm_input_size = hidden_size
            
            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)
        self.forward_layers = forward_layers
        self.backward_layers = backward_layers
    
    def forward(self, inputs, mask):
        batch_size, total_sequence_length = mask.size()
        stacked_sequence_output, final_states, restoration_indices = self.sort_and_run_forward(self._lstm_forward, inputs, mask)
        num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
        if num_valid < batch_size:
            zeros = flow.zeros(
                num_layers, 
                batch_size - num_valid, 
                returned_timesteps, 
                encoder_dim, 
                dtype = stacked_sequence_output.dtype).to(stacked_sequence_output.device)
            zeros = flow.Tensor(zeros)
            stacked_sequence_output = flow.cat([stacked_sequence_output, zeros], 1)
            new_states = []
            for state in final_states:
                state_dim = state.size(-1)
                zeros = state.data.new(num_layers, batch_size - num_valid, state_dim).fill_(0)
                zeros = flow.Tensor(zeros)
                new_states.append(flow.cat([state, zeros], 1))
            final_states = new_states
        
        sequence_length_difference = total_sequence_length - returned_timesteps
        if sequence_length_difference > 0:
            zeros = flow.zeros(
                num_layers, 
                batch_size, 
                sequence_length_difference, 
                stacked_sequence_output[0].size(-1), 
                dtype=stacked_sequence_output.dtype).to(stacked_sequence_output.device)

            zeros = flow.Tensor(zeros)
            stacked_sequence_output = flow.cat([stacked_sequence_output, zeros], 2)
        self._update_states(final_states, restoration_indices)
        return stacked_sequence_output.index_select(1, restoration_indices)
    # TODO: use pytorch instead, modify after oneflow support.
    def _lstm_forward(self, 
                      inputs: PackedSequence,
                      initial_state: Optional[Tuple[flow.Tensor, flow.Tensor]] = None) -> \
        Tuple[flow.Tensor, Tuple[flow.Tensor, flow.Tensor]]:
        
        if initial_state is None:
            hidden_states = [None] * len(self.forward_layers)
        elif initial_state[0].size()[0] != len(self.forward_layers):
            raise Exception("Initial states were passed to forward() but the number of "
                               "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))
        # TODO: use pytorch instead, modify after oneflow support.
        inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        inputs, batch_lengths = flow.tensor(inputs.numpy()), flow.tensor(batch_lengths.numpy())
        if self.use_cuda:
            inputs = inputs.cuda()
            batch_lengths = batch_lengths.cuda()
        forward_output_sequence = inputs
        backward_output_sequence = inputs
        
        final_states = []
        sequence_outputs = []
        for layer_index, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(layer_index))
            backward_layer = getattr(self, 'backward_layer_{}'.format(layer_index))
            
            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence
            
            if state is not None:
                forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
                forward_state = (forward_hidden_state, forward_memory_state)
                backward_state = (backward_hidden_state, backward_memory_state)
            else:
                forward_state = None
                backward_state = None
            
            forward_output_sequence, forward_state = forward_layer(forward_output_sequence,
                                                             batch_lengths,
                                                             forward_state)
            backward_output_sequence, backward_state = backward_layer(backward_output_sequence,
                                                                batch_lengths,
                                                                backward_state)
            if layer_index != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache
            
            sequence_outputs.append(flow.cat([forward_output_sequence,
                                               backward_output_sequence], -1))
            
            final_states.append((flow.cat([forward_state[0], backward_state[0]], -1),
                                 flow.cat([forward_state[1], backward_state[1]], -1)))
            
        stacked_sequence_outputs: flow.Tensor = flow.stack(sequence_outputs)
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple: Tuple[flow.Tensor,
                                 flow.Tensor] = (flow.cat(final_hidden_states, 0),
                                                       flow.cat(final_memory_states, 0))
        return stacked_sequence_outputs, final_state_tuple
            

class LstmbiLm(flow.nn.Module):
    def __init__(self, config, use_cuda=False):
        super(LstmbiLm, self).__init__()
        self.config = config
        self.use_cuda = use_cuda
    
        self.encoder = flow.nn.LSTM(self.config['encoder']['projection_dim'],
                               self.config['encoder']['dim'],
                               num_layers=self.config['encoder']['n_layers'], 
                               bidirectional=True,
                               batch_first=True, 
                               dropout=self.config['dropout'])
        self.projection = flow.nn.Linear(self.config['encoder']['dim'], self.config['encoder']['projection_dim'], bias=True)

    def forward(self, inputs):
        forward, backward = self.encoder(inputs)[0].split(self.config['encoder']['dim'], 2)
        return flow.cat([self.projection(forward), self.projection(backward)], dim=2)