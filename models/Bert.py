import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, args, vocab):
        super(Model, self).__init__()
        self.model_name = 'Bert'
        self.bert = BertModel.from_pretrained(args.pretrain_path)
        self.fc = nn.Linear(args.hidden_dim, args.class_nums)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, labels, attention_mask):
        outputs = self.bert(x, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        output = self.fc(pooled)

        loss = self.loss_func(output, labels)
        return output, loss
