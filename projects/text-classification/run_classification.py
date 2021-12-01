import random
import argparse
from tqdm import tqdm

from projects.optimization import get_schedule
from flowtext.models.bert import bert
from flowtext.models.bert.model_bert import BertForSequenceClassification
from projects.utils import (
    accuracy,
    convert_examples_to_features,
    ColaProcessor,
    MnliProcessor,
    MrpcProcessor
)

import math
import numpy as np
import oneflow as flow
from oneflow.utils.data import (
    TensorDataset, 
    RandomSampler, 
    SequentialSampler, 
    DataLoader
)


processors = {
        "cola": ColaProcessor(),
        "mnli": MnliProcessor(),
        "mrpc": MrpcProcessor(),
}


def set_seed(args):
    if args.seed != None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        flow.manual_seed(args.seed)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type",
                        default=None, 
                        type=str, 
                        required=True,
                        help="Bert pre-trained model type.")
    parser.add_argument("--bert_model",
                        default=None, 
                        type=str, 
                        help="Bert pre-trained model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.0,
                        help="Weight decay to use."
                        )
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps",
                        type=int,
                        default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--lr_scheduler_type",
                        type=str,
                        default="linear",
                        help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', 
                        type=int, 
                        default=13,
                        help="random seed for initialization")
    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    set_seed(args)

    if args.no_cuda:
        device = flow.device("cuda" if flow.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        device = flow.device("cuda")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]
    label_list = processor.get_labels()

    model, tokenizer, _ = bert(
        pretrained=True, 
        model_type=args.model_type, 
        checkpoint_path=args.bert_model, 
        bert_type=BertForSequenceClassification
    )
    tokenizer.do_lower_case = args.do_lower_case

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
    model.to(device)

    defaults = {
            "lr": args.learning_rate,
            "clip_grad_max_norm": 1.0,
            "clip_grad_norm_type": 2.0,
        }
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            **defaults
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            **defaults
        }
    ]
    
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer
            )
        all_input_ids = flow.tensor([f.input_ids for f in train_features], dtype=flow.long)
        all_input_mask = flow.tensor([f.input_mask for f in train_features], dtype=flow.long)
        all_segment_ids = flow.tensor([f.segment_ids for f in train_features], dtype=flow.long)
        all_label_ids = flow.tensor([f.label_id for f in train_features], dtype=flow.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = args.train_batch_size)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        
        optimizer = flow.optim.AdamW(optimizer_grouped_parameters)
        lr_schedule = get_schedule(
            name = args.lr_scheduler_type,
            optimizer = optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
        global_step = 0
        for epoch in range(int(args.num_train_epochs)):
            tr_loss, nb_tr_steps = 0, 0
            model.train()
            for step, batch in tqdm(enumerate(train_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss, _ = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_steps += 1
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_schedule.step()
                    optimizer.zero_grad()
                    global_step += 1
                if global_step >= args.max_train_steps:
                    break

    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer
            )
        all_input_ids = flow.tensor([f.input_ids for f in eval_features], dtype=flow.long)
        all_input_mask = flow.tensor([f.input_mask for f in eval_features], dtype=flow.long)
        all_segment_ids = flow.tensor([f.segment_ids for f in eval_features], dtype=flow.long)
        all_label_ids = flow.tensor([f.label_id for f in eval_features], dtype=flow.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with flow.no_grad():
                tmp_eval_loss, logits = model(
                    input_ids=input_ids, 
                    attention_mask=input_mask, 
                    token_type_ids=segment_ids, 
                    labels=label_ids
                    )
            
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': tr_loss/nb_tr_steps}


if __name__ == '__main__':
    main()
