from tqdm import tqdm
from loguru import logger
import numpy as np
from scipy.stats import spearmanr

import oneflow as flow


def cosine_similarity(x, y, dim=-1):
    return (
        flow.sum(x * y, dim=dim)
        / (flow.linalg.norm(x, dim=dim) * flow.linalg.norm(y, dim=dim))
    )


def eval(model, dataloader, device):
    model.eval()

    sim_tensor = flow.tensor([], device=device)
    label_array = np.array([])

    with flow.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
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
    
    model.train()
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def train(model, train_dataloader, dev_dataloader, lr, best_score, early_stop, device, save_path):
    model.train()
    if early_stop:
        early_stop_step = 0
    optimizer = flow.optim.AdamW(model.parameters(), lr=lr)

    for step, batch in enumerate(tqdm(train_dataloader), start=1):
        size = batch['input_ids'].size()
        bs, num_sent = size[0], size[1]

        input_ids = batch['input_ids'].view(bs * num_sent, -1).to(device)
        attention_mask = batch['attention_mask'].view(bs * num_sent, -1).to(device)

        loss = model(input_ids, attention_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = eval(model, dev_dataloader, device)
            
            if best_score < corrcoef:
                if early_stop:
                    early_stop_step = 0 
                best_score = corrcoef
                flow.save(model.state_dict(), save_path)
                logger.info(f"higher corrcoef: {best_score:.4f} in batch: {step}, save model")
                continue

            if early_stop:
                early_stop_step += 1
                if early_stop_step == 30:
                    logger.info(f"corrcoef doesn't improve for {early_stop_step} batch, early stop!")
                    logger.info(f"train use sample number: {(step - 10) * bs}")
                    return 