import oneflow as flow
from oneflow import nn
from .highway import Highway
import logging

logger = logging.getLogger('elmoformanylangs')


class EmbeddingLayer(nn.Module):
    def __init__(self, 
                 n_d, 
                 word2id, 
                 embs=None, 
                 fix_emb=True, 
                 oov='<oov>', 
                 pad='<pad>', 
                 normalize=True):
        super().__init__()
        if embs is not None:
            embwords, embvecs = embs
            logger.info("{} pre-trained word embeddings loaded.".format(len(word2id)))
            if n_d != len(embvecs[0]):
                logger.warning("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.".format(
                                                                    n_d, len(embvecs[0]), len(embvecs[0]))) 
                n_d = len(embvecs[0])
                
        self.word2id = word2id
        self.id2word = {i:word for word, i in word2id.items()}
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d, padding_idx=self.padid)
        self.embedding.weight.data.uniform_(-0.25, 0.25)
        
        if embs is not None:
            weight = self.embedding.weight
            weight.data[:len(embwords)].copy_(flow.tensor(embvecs))
            logger.info("embedding shape: {}".format(weight.size()))
        
        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2, 1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.copy_(weight.data.div(norms.expand_as(weight.data)))
            
        if fix_emb:
            self.embedding.weight.requires_grad = False
    
    def forward(self, input_):  
        return self.embedding(input_)


class LstmTokenEmbedder(nn.Module):
    def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda=False):
        super().__init__()
        self.config = config
        self.use_cuda = use_cuda
        self.word_emb_layer = word_emb_layer
        self.char_emb_layer = char_emb_layer
        self.output_dim = config['encoder']['projection_dim']
        emb_dim = 0
        
        if word_emb_layer is not None:
            emb_dim += word_emb_layer.n_d
        
        if char_emb_layer is not None:
            emb_dim += char_emb_layer.n_d * 2
            self.char_lstm = nn.LSTM(char_emb_layer.n_d, char_emb_layer.n_d, num_layers=1, bidirectional=True,
                                     batch_first=True, dropout=config['dropout'])
            self.projection = nn.Linear(emb_dim, self.output_dim, bias=True)
            
    def forward(self, word_inp, chars_inp, shape):
        embs = []
        batch_size, seq_len = shape
        if self.word_emb_layer is not None:
            word_emb = self.word_emb_layer(flow.Tensor(word_inp).long().cuda() if self.use_cuda else flow.Tensor(word_inp).long())
            embs.append(word_emb)
        
        if self.char_emb_layer is not None:
            chars_inp = chars_inp.view(batch_size * seq_len, -1)
            chars_emb = self.char_emb_layer(flow.Tensor(chars_inp).long().cuda() if self.use_cuda else flow.Tensor(chars_inp).long())
            _, (chars_outputs, _) = self.char_lstm(chars_emb)
            chars_outputs = chars_outputs.contiguous().view(-1, self.config['token_embedder']['char_dim'] * 2)
            embs.append(chars_outputs)
            
        token_embedding = flow.cat(embs, dim=2)
        
        return self.projection(token_embedding)


class ConvTokenEmbedder(nn.Module):
    def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda):
        super().__init__()
        self.config = config
        self.use_cuda = use_cuda
        
        self.word_emb_layer = word_emb_layer
        self.char_emb_layer = char_emb_layer
        
        self.output_dim = config['encoder']['projection_dim']
        self.emb_dim = 0
        if word_emb_layer is not None:
            self.emb_dim += word_emb_layer.n_d
        
        if char_emb_layer is not None:
            self.convolutions = nn.ModuleList()
            cnn_config = config['token_embedder']
            filters = cnn_config['filters']
            char_embed_dim = cnn_config['char_dim']
            
            for i, (width, num) in enumerate(filters):
                conv = flow.nn.Conv1d(
                    in_channels=char_embed_dim,
                    out_channels=num,
                    kernel_size=width,
                    bias=True
                )
                self.convolutions.append(conv)
            self.n_filters = sum(f[1] for f in filters)
            self.n_highway = cnn_config['n_highway']
            self.highways = Highway(self.n_filters, self.n_highway, activation=flow.nn.functional.relu)
            self.emb_dim += self.n_filters
            
        self.projection = nn.Linear(self.emb_dim, self.output_dim, bias=True)
    
    def forward(self, word_inp, chars_inp, shape):
        embs = []
        batch_size, seq_len = shape
        if self.word_emb_layer is not None:
            batch_size, seq_len = word_inp.size(0), word_inp.size(1)
            word_emb = self.word_emb_layer(flow.Tensor(word_inp).long().cuda() if self.use_cuda else flow.Tensor(word_inp).long())
            embs.append(word_emb)
        
        if self.char_emb_layer is not None:
            chars_inp = chars_inp.view(batch_size * seq_len, -1)
            character_embedding = self.char_emb_layer(flow.Tensor(chars_inp).long().cuda() if self.use_cuda else flow.Tensor(chars_inp).long())
            character_embedding = flow.transpose(character_embedding, 1, 2)
            
            cnn_config = self.config['token_embedder']
            if cnn_config['activation'] == 'tanh':
                activation = flow.nn.functional.tanh
            elif cnn_config['activation'] == 'relu':
                activation = flow.nn.functional.relu
            else:
                raise Exception("Unknown activation")
            
            convs = []
            for i in range(len(self.convolutions)):
                convolved = self.convolutions[i](character_embedding)
                convolved, _ = flow.max(convolved, dim=-1)
                convolved = activation(convolved)
                convs.append(convolved)
            char_emb = flow.cat(convs, dim=-1)
            char_emb = self.highways(char_emb)
            
            embs.append(char_emb.view(batch_size, -1, self.n_filters))
        token_embedding = flow.cat(embs, dim=2)
        return self.projection(token_embedding)