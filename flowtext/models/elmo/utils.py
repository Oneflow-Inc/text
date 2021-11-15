import oneflow as flow
from oneflow import Tensor
import collections
import random
import logging


logger = logging.getLogger('elmoformanylangs')

def recover(li, ind):
    dummy = list(range(len(ind)))
    dummy.sort(key=lambda l: ind[l])
    li = [li[i] for i in dummy]
    return li


def get_lengths_from_binary_sequence_mask(mask: flow.Tensor):
    return mask.long().sum(-1)


def sort_batch_by_length(tensor, sequence_lengths):
    if not isinstance(tensor, Tensor) or not isinstance(sequence_lengths, Tensor):
        raise Exception("Both the tensor and sequence length must be flow.Tensor.")
    (sorted_sequence_lengths, permutation_index) = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    sequence_lengths.data.copy_(flow.arange(0, sequence_lengths.size(0)))
    index_range = sequence_lengths.clone()
    index_range = flow.Tensor(index_range.long())
    
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index
# TODO: modify after the orthogonal supported.
# def block_orthogonal(tensor: flow.Tensor, split_sizes: List[int], gain: float = 1.0) -> None:
#     if isinstance(tensor, Tensor):
#         sizes = list(tensor.size())
#         if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
#             raise Exception("tensor dimensions must be divisible by their respective "
#                                          "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
#         indexes = [list(range(0, max_size, split))
#                        for max_size, split in zip(sizes, split_sizes)]
#         for block_start_indices in itertools.product(*indexes):
#             index_and_step_tuples = zip(block_start_indices, split_sizes)
#             block_slice = tuple([slice(start_index, start_index + step)
#                                  for start_index, step in index_and_step_tuples])
#             tensor[block_slice] = flow.nn.init.orthogonal_(tensor[block_slice].contiguous(), gain=gain)


def get_dropout_mask(dropout_probability: float, tensor_for_masking: Tensor):
    binary_mask = tensor_for_masking.clone()
    binary_mask.data.copy_(flow.rand(tensor_for_masking.size()) > dropout_probability)

    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


def dict2namedtuple(dic):
    return collections.namedtuple('Namespace', dic.keys())(**dic)


def read_list(sents, max_chars=None):
    dataset = []
    textset = []
    for sent in sents:
        data = ['<bos>']
        text = []
        for token in sent:
            text.append(token)
            if max_chars is not None and len(token) + 2 > max_chars:
                token = token[:max_chars - 2]
            data.append(token)
        data.append('<eos>')
        dataset.append(data)
        textset.append(text)
    return dataset, textset

def create_one_batch(x, word2id, char2id, config, oov='<oov>', pad='<pad>', sort=True):
    batch_size = len(x)
    lst = list(range(batch_size))
    if sort:
        lst.sort(key=lambda l: -len(x[l]))
    x = [x[i] for i in lst]
    lens = [len(x[i]) for i in lst]
    max_len = max(lens)
    if word2id is not None:
        oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
        assert oov_id is not None and pad_id is not None
        batch_w = flow.zeros(batch_size, max_len).fill_(pad_id)
        for i, x_i in enumerate(x):
            for j, x_ij in enumerate(x_i):
                batch_w[i, j] = word2id.get(x_ij, oov_id)
        batch_w = batch_w.long()
    else:
        batch_w = None
    if char2id is not None:
        bow_id, eow_id, oov_id, pad_id = [char2id.get(key, None) for key in ('<eow>', '<bow>', oov, pad)]
        assert bow_id is not None and eow_id is not None and oov_id is not None and pad_id is not None
        if config['token_embedder']['name'].lower() == 'cnn':
            max_chars = config['token_embedder']['max_characters_per_token']
            assert max([len(w) for i in lst for w in x[i]]) + 2 <= max_chars
        elif config['token_embedder']['name'].lower() == 'lstm':
            max_chars = max([len(w) for i in lst for w in x[i]]) + 2
        else:
            raise ValueError('Unknown token_embedder: {0}'.format(config['token_embedder']['name']))
        
        batch_c = flow.zeros(batch_size, max_len, max_chars).fill_(pad_id)
        for i, x_i in enumerate(x):
            for j, x_ij in enumerate(x_i):
                batch_c[i, j, 0] = bow_id
                if x_ij == '<bos>' or x_ij == '<eos>':
                    batch_c[i, j, 1] = char2id.get(x_ij)
                    batch_c[i, j, 2] = eow_id
                else:
                    for k, c in enumerate(x_ij):
                        batch_c[i, j, k + 1] = char2id.get(c, oov_id)
                    batch_c[i, j, len(x_ij) + 1] = eow_id
        batch_c = batch_c.long()
    else:
        batch_c = None
    masks = [flow.zeros(batch_size, max_len), [], []]
    for i, x_i in enumerate(x):
        for j in range(len(x_i)):
            masks[0][i, j] = 1
            if j + 1 < len(x_i):
                masks[1].append(i * max_len + j)
            if j > 0:
                masks[2].append(i * max_len + j)
    assert len(masks[1]) <= batch_size * max_len
    assert len(masks[2]) <= batch_size * max_len
    
    masks[0] = flow.Tensor(masks[0]).long()
    masks[1] = flow.Tensor(masks[1]).long()
    masks[2] = flow.Tensor(masks[2]).long()
    return batch_w, batch_c, lens, masks
        

def create_batches(x, batch_size, word2id, char2id, config, perm=None, shuffle=False, sort=True, text=None):
    ind = list(range(len(x)))
    lst = perm or list(range(len(x)))
    if shuffle:
        random.shuffle(lst)

    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    x = [x[i] for i in lst]
    ind = [ind[i] for i in lst]
    if text is not None:
        text = [text[i] for i in lst]

    sum_len = 0.0
    batches_w, batches_c, batches_lens, batches_masks, batches_text, batches_ind = [], [], [], [], [], []
    size = batch_size
    nbatch = (len(x) - 1) // size + 1
    for i in range(nbatch):
        start_id, end_id = i * size, (i + 1) * size
        bw, bc, blens, bmasks = create_one_batch(x[start_id: end_id], word2id, char2id, config, sort=sort)
        sum_len += sum(blens)
        batches_w.append(bw)
        batches_c.append(bc)
        batches_lens.append(blens)
        batches_masks.append(bmasks)
        batches_ind.append(ind[start_id: end_id])
        if text is not None:
            batches_text.append(text[start_id: end_id])

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_w = [batches_w[i] for i in perm]
        batches_c = [batches_c[i] for i in perm]
        batches_lens = [batches_lens[i] for i in perm]
        batches_masks = [batches_masks[i] for i in perm]
        batches_ind = [batches_ind[i] for i in perm]
        if text is not None:
            batches_text = [batches_text[i] for i in perm]

    logger.info("{} batches, avg len: {:.1f}".format(
        nbatch, sum_len / len(x)))
    recover_ind = [item for sublist in batches_ind for item in sublist]
    if text is not None:
        return batches_w, batches_c, batches_lens, batches_masks, batches_text, recover_ind
    return batches_w, batches_c, batches_lens, batches_masks, recover_ind