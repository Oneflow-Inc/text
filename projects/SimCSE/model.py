import oneflow.nn as nn
import oneflow as flow


def cosine_similarity(x, y, dim=-1):
    return (
        flow.sum(x * y, dim=dim)
        / (flow.linalg.norm(x, dim=dim) * flow.linalg.norm(y, dim=dim))
    )


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.Tanh()
    
    def forward(self, hidden):
        x = self.dense(hidden)
        x = self.act(x)
        return x


class Simcse(nn.Module):
    def __init__(self, bert, task, pooler_type='cls'):
        super().__init__()
        self.bert = bert
        hidden_size = self.bert.config.hidden_size
        self.mlp = MLP(hidden_size)
        self.task = task
        self.pooler_type = pooler_type

        assert self.task in ['sup', 'unsup']
    
    def pooler(self, inputs, attention_mask):
        if self.pooler_type == 'cls':
            return inputs[0][:, 0]
        
        elif self.pooler_type == 'pooled':
            return inputs[1]
        
        elif self.pooler_type == 'last-avg':
            last_hidden = inputs[0].permute(0, 2, 1)
            # print(last_hidden.size())
            # print(attention_mask.unsqueeze(-1).size())
            last_hidden = last_hidden * attention_mask.unsqueeze(1)
            return nn.AvgPool1d(kernel_size=last_hidden.size(-1))(last_hidden).squeeze(-1)

        elif self.pooler_type == 'first-last-avg':
            first_hidden = inputs[2][1]
            last_hidden = inputs[0]
            res = (first_hidden + last_hidden) * attention_mask.unsqueeze(-1)
            res = res.permute(0, 2, 1)
            return nn.AvgPool1d(kernel_size=res.size(-1))(res).squeeze(-1)

    def loss_unsup(self, y_pred):
        y_true = flow.arange(y_pred.shape[0], device=y_pred.device)
        y_true = (y_true - y_true % 2 * 2) + 1
        sim = cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0))
        sim = sim - flow.eye(y_pred.shape[0], device=y_pred.device) * 1e12
        sim = sim / 0.05
        loss = nn.CrossEntropyLoss()(sim, y_true)
        return loss

    def loss_sup(self, y_pred):
        y_true = flow.arange(y_pred.size(0), device=y_pred.device)
        use_row = flow.where((y_true + 1) % 3 != 0)[0]
        y_true = (use_row - use_row % 3 * 2) + 1
        sim = cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0))
        sim = sim - flow.eye(y_pred.shape[0], device=y_pred.device) * 1e12
        sim = flow.index_select(sim, 0, use_row)
        sim = sim / 0.05
        loss = nn.CrossEntropyLoss()(sim, y_true)
        return loss

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        out = self.pooler(out, attention_mask)
        if self.task == "unsup":
            out = self.mlp(out)

        if self.training:
            if self.task == 'sup':
                loss = self.loss_sup(out)
            else:
                loss = self.loss_unsup(out)
            return loss

        return out