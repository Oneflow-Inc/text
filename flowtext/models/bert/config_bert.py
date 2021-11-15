import json

from oneflow import nn


class BertConfig(object):
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act=nn.GELU(),
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        classifier_dropout=None,
        is_decoder=False,
        add_cross_attention=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.classifier_dropout = classifier_dropout
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

    def load_from_json(self, config_path):
        with open(config_path, "r", encoding="utf8") as file:
            json_data = json.load(file)
            for k, v in json_data.items():
                if hasattr(self, k):
                    setattr(self, k, v)
