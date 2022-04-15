import sys, os
path = '/'.join(os.getcwd().split('/')[:-2])
sys.path.append(path)

import argparse
from loguru import logger

from model import Simcse
from dataset import load_data, TrainDataset, TestDataset
from trainer import train, eval
from flowtext.models.bert import bert as BertModel
import oneflow as flow
from oneflow.utils.data import DataLoader


def get_argparse():
    parser = argparse.ArgumentParser("SimCSE")
    parser.add_argument("--task", type=str, default='sup', help='The type of task, sup or unsup.')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=64, help='The max len of sequence.')
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooler_type", type=str, default='cls')
    parser.add_argument("--early_stop", default=False, action='store_true')
    parser.add_argument("--model_type", type=str, default="bert-base-chinese", help='The model_type of bert.')
    parser.add_argument("--pretrained_path", type=str, help='The path of pretrained model.')
    parser.add_argument("--save_path", type=str, default='./saved_simcse', help='save checkpoint path.')
    parser.add_argument("--train_data_path", type=str, help='The path of train dataset.')
    parser.add_argument("--train_data_path2", type=str, default=None,  help='The path of train dataset.')
    parser.add_argument("--dev_data_path", type=str, default=None,  help='The path of dev dataset.')
    parser.add_argument("--test_data_path", type=str, help='The path of test dataset.')
    return parser.parse_args()

def main(args):
    device = args.device
    epochs = args.epochs
    batch_size = args.batch_size
    dropout = args.dropout
    model_type = args.model_type
    pretrained_path = args.pretrained_path
    train_data_path = args.train_data_path
    train_data_path2 = args.train_data_path2
    dev_data_path = args.dev_data_path
    test_data_path = args.test_data_path
    train_data_name = 'snli' + '-' + args.task
    dev_data_name = 'sts'
    test_data_name = 'sts'

    logger.info(f'device: {device}, model path: {pretrained_path}')
    if not os.path.exists(pretrained_path):
        pretrained_path = None
    bert, tokenizer, _ = BertModel(pretrained=True, model_type=model_type, checkpoint_path=pretrained_path)
    bert.hidden_dropout_prob = dropout
    bert.attention_probs_dropout_prob = dropout

    train_dataloader = DataLoader(TrainDataset(
        train_data_name, 
        train_data_path, 
        tokenizer, 
        max_len=args.max_len,
        task=args.task, 
        name2='sts', 
        path2=train_data_path2
        ), batch_size=batch_size)

    dev_dataloader = DataLoader(TestDataset(dev_data_name, dev_data_path, tokenizer, max_len=args.max_len), batch_size=batch_size)
    test_dataloader = DataLoader(TestDataset(test_data_name, test_data_path, tokenizer, max_len=args.max_len), batch_size=batch_size)
    
    
    model = Simcse(bert, task=args.task, pooler_type=args.pooler_type)
    model.to(device)
    optimizer = flow.optim.AdamW(model.parameters(), lr=args.lr)

    best_score = 0
    for epoch in range(epochs):
        logger.info(f'epoch: {epoch}')
        train(
            model, 
            train_dataloader, 
            dev_dataloader, 
            args.lr,
            best_score,
            early_stop=args.early_stop, 
            device=device, 
            save_path=args.save_path
            )
    logger.info(f'train is finished, best model is saved at {args.save_path}')
    model.load_state_dict(flow.load(args.save_path))
    dev_corrcoef = eval(model, dev_dataloader, device)
    test_corrcoef = eval(model, test_dataloader, device)
    logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
    logger.info(f'test_corrcoef: {test_corrcoef:.4f}')


if __name__ == '__main__':
    args = get_argparse()
    main(args)