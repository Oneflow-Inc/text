import sys, os
sys.path.append('/home/xiezipeng/text')

import random
import time
from typing import Dict, List

import jsonlines
import numpy as np
import oneflow as flow
import oneflow.nn as nn
from loguru import logger
from scipy.stats import spearmanr
from oneflow.utils.data import DataLoader, Dataset
from tqdm import tqdm
from flowtext.models.bert import bert


EPOCHS = 1
SAMPLES = 10000
BATCH_SIZE = 64
LR = 1e-5
DROPOUT = 0.3
MAXLEN = 64
DEVICE = flow.device('cuda' if flow.cuda.is_available() else 'cpu') 

# 预训练模型目录
BERT = '/home/xiezipeng/text/pretrained_flow/bert-base-chinese-oneflow'
BERT_WWM_EXT = 'pretrained_model/bert_wwm_ext_pytorch'
ROBERTA = 'pretrained_model/roberta_wwm_ext_pytorch'
model_path = BERT 

# 微调后参数存放位置
SAVE_PATH = './saved_model/simcse_unsup'

# 数据目录
SNIL_TRAIN = './datasets/SNLI/train.txt'
STS_TRAIN = './datasets/STS/cnsd-sts-train.txt'
STS_DEV = './datasets/STS/cnsd-sts-dev.txt'
STS_TEST = './datasets/STS/cnsd-sts-test.txt'


def load_data(name: str, path: str) -> List:
    def load_snli_data(path):
        with jsonlines.open(path, 'r') as f:
            return [line.get('origin') for line in f]

    def load_lqcmc_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [line.strip().split('\t')[0] for line in f]    

    def load_sts_data(path):
        with open(path, 'r', encoding='utf8') as f:            
            return [(line.split("||")[1], line.split("||")[2], line.split("||")[3]) for line in f]

    assert name in ["snli", "lqcmc", "sts"]
    if name == 'snli':
        return load_snli_data(path)
    return load_lqcmc_data(path) if name == 'lqcmc' else load_sts_data(path) 


class TrainDataset(Dataset):
    def __init__(self, name, path, tokenizer, max_len=MAXLEN, name2=None, path2=None):
        self.data = load_data(name, path)
        if name2 is not None and path2 is not None:
            data2 = load_data(name2, path2)
            data2 = [i[0] for i in data2]
            self.data = self.data + data2
            random.shuffle(self.data)
            # self.data = random.sample(self.data, SAMPLES)
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

    def __getitem__(self, index):
        ids, mask = self.text2id(self.data[index])
        return {
            "input_ids" : flow.tensor([ids, ids], dtype=flow.long),
            "attention_mask" : flow.tensor([mask, mask], dtype=flow.long)
        }

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


class Simcse(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids, attention_mask, token_type_ids)
        return out[0][:, 0]


def cosine_similarity(x, y, dim=-1):
    return (
        flow.sum(x * y, dim=dim)
        / (flow.linalg.norm(x, dim=dim) * flow.linalg.norm(y, dim=dim))
    )

    
def loss_unsup(y_pred):
    y_true = flow.arange(y_pred.shape[0], device=DEVICE)
    y_true = (y_true - y_true % 2 * 2) + 1
    sim = cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0))
    sim = sim - flow.eye(y_pred.shape[0], device=DEVICE) * 1e12
    sim = sim / 0.05
    loss = nn.CrossEntropyLoss()(sim, y_true)
    return loss


def eval(model, dataloader):
    model.eval()
    sim_tensor = flow.tensor([], device=DEVICE)
    label_array = np.array([])
    with flow.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].numpy()

            sent1_input_ids = input_ids[:,0]
            sent1_attention_mask = attention_mask[:,0]
            sent1_res = model(sent1_input_ids, sent1_attention_mask)

            sent2_input_ids = input_ids[:,1]
            sent2_attention_mask = attention_mask[:,1]
            sent2_res = model(sent2_input_ids, sent2_attention_mask)

            sim = cosine_similarity(sent1_res, sent2_res)
            sim_tensor = flow.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(labels))
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation

            
def train(model, train_dataloader, dev_dataloader, optimizer):
    model.train()
    global best
    early_stop_step = 0
    for step, batch in enumerate(tqdm(train_dataloader), start=1):
        bs = batch['input_ids'].size(0)
        input_ids = batch['input_ids'].view(bs * 2, -1).to(DEVICE)
        attention_mask = batch['attention_mask'].view(bs * 2, -1).to(DEVICE)

        out = model(input_ids, attention_mask)
        loss = loss_unsup(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = eval(model, dev_dataloader)
            model.train()
            if best < corrcoef:
                early_stop_step = 0 
                best = corrcoef
                flow.save(model.state_dict(), SAVE_PATH)
                logger.info(f"higher corrcoef: {best:.4f} in batch: {step}, save model")
                continue
            early_stop_step += 1
            if early_stop_step == 2000:
                logger.info(f"corrcoef doesn't improve for {early_stop_step} batch, early stop!")
                logger.info(f"train use sample number: {(step - 10) * BATCH_SIZE}")
                return 
            

if __name__ == '__main__':
    logger.info(f'device: {DEVICE}, model path: {model_path}')
    bert, tokenizer, _ = bert(pretrained=True, model_type='bert-base-chinese', checkpoint_path=model_path)
    bert.hidden_dropout_prob=DROPOUT
    bert.attention_probs_dropout_prob=DROPOUT
    train_dataloader = DataLoader(TrainDataset('snli', SNIL_TRAIN, tokenizer, name2='sts', path2=STS_TRAIN), batch_size=BATCH_SIZE)
    dev_dataloader = DataLoader(TestDataset('sts', STS_DEV, tokenizer), batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(TestDataset('sts', STS_TEST, tokenizer), batch_size=BATCH_SIZE)
    
    model = Simcse(bert)
    model.to(DEVICE)
    optimizer = flow.optim.AdamW(model.parameters(), lr=LR)

    best=0
    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, dev_dataloader, optimizer)
    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')
    model.load_state_dict(flow.load(SAVE_PATH))
    dev_corrcoef = eval(model, dev_dataloader)
    test_corrcoef = eval(model, test_dataloader)
    logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
    logger.info(f'test_corrcoef: {test_corrcoef:.4f}')