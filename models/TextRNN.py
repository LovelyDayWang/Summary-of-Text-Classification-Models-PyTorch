import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, vocab):
        super(Model, self).__init__()
        self.model_name = 'TextRNN'

        self.embedding = nn.Embedding(args.vocab_nums, args.embed_size, padding_idx=0)
        self.dropout = nn.Dropout(args.dropout)

        self.bi_lstm = nn.LSTM(args.embed_size, args.hidden_dim, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(args.hidden_dim * 2, args.class_nums)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, labels, mask):
        x_embed = self.embedding(x)
        x_embed = self.dropout(x_embed)

        out, _ = self.bi_lstm(x_embed)
        out = self.fc(out[:, -1, :])

        loss = self.loss_func(out, labels).cpu()

        return out, loss
