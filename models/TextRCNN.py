import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, vocab):
        super(Model, self).__init__()
        self.model_name = 'TextRCNN'

        self.embedding = nn.Embedding(args.vocab_nums, args.embed_size, padding_idx=0)

        self.lstm = nn.LSTM(args.embed_size, args.hidden_dim,
                            bidirectional=True, batch_first=True, dropout=args.dropout)
        self.maxpool = nn.MaxPool1d(args.pad_size)
        self.fc = nn.Linear(args.hidden_dim * 2 + args.embed_size, args.class_nums)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, labels, masks):
        x_embed = self.embedding(x)
        out, _ = self.lstm(x_embed)
        out = torch.cat((x_embed, out), 2)
        out = torch.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)

        loss = self.loss_func(out, labels)

        return out, loss

