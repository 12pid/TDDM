import torch
import torch.nn as nn
import mxnet as mx


class Enh(nn.Module):
    def __init__(self, input_dim, num_classes, T, lam, fa, ff):
        super(Enh, self).__init__()
        self.T = T
        self.fa = fa
        self.ff = ff
        self.lam = lam
        self.head = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x (B d H W)
        score = self.head(x) / torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0,1)
        score = score.flatten(2)
        base_logit = torch.mean(score, dim=2)

        if self.T == 99:
            att_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=2)
        fusion = base_logit * att_logit
        return base_logit + self.fa * att_logit + self.ff * fusion


class FEM(nn.Module):
    temp_settings = {
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }
    def __init__(self, num_heads, lam, input_dim, num_classes):
        super(FEM, self).__init__()
        self.temp_list = self.temp_settings[num_heads] # [1]
        self.multi_head = nn.ModuleList([
            Enh(input_dim, num_classes, self.temp_list[i], lam)
            for i in range(num_heads)
        ])

    def forward(self, x):
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        return logit