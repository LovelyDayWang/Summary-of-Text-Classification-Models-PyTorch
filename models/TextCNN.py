import torch
import torch.nn as nn
from math import floor
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args, vocab):
        super(Model, self).__init__()
        self.model_name = 'TextCNN'  # Multi scale CNN

        self.embedding = nn.Embedding(args.vocab_nums, args.embed_size, padding_idx=0)

        filter_sizes = args.filter_size.split(',')
        self.filter_num = len(filter_sizes)
        self.convs = nn.ModuleList()

        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            tmp = nn.Conv1d(args.embed_size, args.filter_map_nums, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            self.convs.add_module('conv-{}'.format(filter_size), tmp)

        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(self.filter_num * args.filter_map_nums, args.class_nums)

        self.loss_func = nn.CrossEntropyLoss()

    def conv_and_pool(self, x, conv):
        x = torch.relu(conv(x))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, labels, mask):
        x_embed = self.embedding(x)

        x_embed = x_embed.transpose(1, 2)
        conv_result = []
        for tmp in self.convs:
            out = self.conv_and_pool(x_embed, tmp)
            conv_result.append(out)

        out = torch.cat(conv_result, dim=1)
        out = self.dropout(out)
        out = self.fc(out)

        loss = self.loss_func(out, labels).cpu()

        return out, loss
