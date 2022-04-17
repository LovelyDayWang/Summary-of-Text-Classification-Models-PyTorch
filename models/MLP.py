import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args, vocab):
        super(Model, self).__init__()
        self.model_name = 'MLP'
        self.embedding = nn.Embedding(args.vocab_nums, args.embed_size, padding_idx=0)
        self.linear1 = nn.Linear(args.embed_size, args.hidden_dim)
        self.linear2 = nn.Linear(args.hidden_dim, args.class_nums)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, labels, mask):
        x_embed = self.embedding(x)

        out = self.linear1(x_embed.sum(dim=1))
        out = F.relu(out)
        out = self.linear2(out)

        loss = self.loss_func(out, labels)
        return out, loss
