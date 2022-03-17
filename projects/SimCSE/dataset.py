from typing import Dict, List

import oneflow as flow
from oneflow.utils.data import DataLoader, Dataset


def load_data(name: str, path: str) -> List:
    def load_snli_data_unsup(path):
        with jsonlines.open(path, 'r') as f:
            return [line.get('origin') for line in f]

    def load_snli_data_sup(path):        
        with jsonlines.open(path, 'r') as f:
            return [(line['origin'], line['entailment'], line['contradiction']) for line in f]

    def load_lqcmc_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [line.strip().split('\t')[0] for line in f]    

    def load_sts_data(path):
        with open(path, 'r', encoding='utf8') as f:            
            return [(line.split("||")[1], line.split("||")[2], line.split("||")[3]) for line in f]

    assert name in ["snli-sup", "snli-unsup", "lqcmc", "sts"]
    if name == 'snli-sup':
        return load_snli_data_sup(path)
    elif name == 'snli-unsup':
        return load_snli_data_unsup(path)
    elif name == 'sts':
        return load_sts_data(path)
    else:
        return load_lqcmc_data(path)


class TrainDataset(Dataset):
    def __init__(self, name, path, tokenizer, task, max_len, name2='sts', path2=None):
        self.task = task
        self.data = load_data(name, path)
        if path2 is not None:
            data2 = load_data(='sts', path2)
            data2 = [i[0] for i in data2]
            self.data = self.data + data2
        if name == 'snli':
            random.shuffle(self.data)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
    
    def __len__(self):
        return len(self.data)

    def pad_text(self, ids):
        attention_mask = [1] * len(ids)
        ids = ids + [self.pad_id] * (self.max_len - len(ids))
        attention_mask = attention_mask + [self.pad_id] * (self.max_len - len(attention_mask))
        return ids, attention_mask
    
    def text2id(self, text):
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids[: self.max_len - 2]
        ids = [self.cls_id] + ids + [self.sep_id]
        ids, attention_mask = self.pad_text(ids)
        return ids, attention_mask

    def supervise_task(self, index):
        ids0, mask0 = self.text2id(self.data[index][0])
        ids1, mask1 = self.text2id(self.data[index][1])
        ids2, mask2 = self.text2id(self.data[index][2])
        return {
            "input_ids" : flow.tensor([ids0, ids1, ids2], dtype=flow.long),
            "attention_mask" : flow.tensor([mask0, mask1, mask2], dtype=flow.long)
        }
    
    def unsupervised_task(self, index):
        ids, mask = self.text2id(self.data[index])
        return {
            "input_ids" : flow.tensor([ids, ids], dtype=flow.long),
            "attention_mask" : flow.tensor([mask, mask], dtype=flow.long)
        }

    def __getitem__(self, index):
        if self.task == "sup":
            return self.supervise_task(index)
        elif self.task == "unsup":
            return self.unsupervise_task(index)


class TestDataset(TrainDataset):
    def __getitem__(self, index):
        label = int(self.data[index][2])
        ids0, mask0 = self.text2id(self.data[index][0])
        ids1, mask1 = self.text2id(self.data[index][1])
        return {
            "input_ids" : flow.tensor([ids0, ids1], dtype=flow.long),
            "attention_mask" : flow.tensor([mask0, mask1], dtype=flow.long),
            "labels" : label
        }
