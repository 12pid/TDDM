from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
from .fun import Enh, FEM
import torch.utils.model_zoo as model_zoo
import logging
import dlib
import math
from .util_gcn import *
import torch
from apex import amp
from torch.nn import Parameter, Linear
import torch.nn as nn
import torch.nn.functional as F
import pickle

device = torch.device("cuda:0")

with open('data/voc/voc_glove_word2vec.pkl', 'rb') as f:
    inp = pickle.load(f)
inpt = torch.Tensor(inp)
inp_var =inpt.float().detach()

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        detector = dlib.get_frontal_face_detector()#
        input = input.to(device)
        self.weight = self.weight.to(device)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Max_attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,  #8
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Max_attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.pool = nn.MaxPool1d(256)

    def forward(self, x):
        N, C = x.shape
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        Pq = self.pool(q)
        Pq = Pq.repeat(1, 1, 20)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + 0.1 * Pq
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ResNet_FEM(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, num_heads, lam, num_classes, depth=101, input_dim=2048):
        self.block, self.layers = self.arch_settings[depth]
        self.depth = depth
        super(ResNet, self).__init__(self.block, self.layers)
        self.classifier = FEM(num_heads, lam, input_dim, num_classes)
        self.loss_func = F.binary_cross_entropy_with_logits

        #GCN
        _adj = gen_A(num_classes, t=0.4, adj_file='data/voc/voc_adj.pkl')
        self.A = Parameter(torch.from_numpy(_adj).float())
        self.gc1 = GraphConvolution(300, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.poolingtrans = nn.MaxPool1d(14 * 14)
        self.atten = Max_attention(dim=2048)

        #trans
        self.transform_14 = nn.Conv2d(2048, 512, 1)
        self.transform_28 = nn.Conv2d(2048 // 2, 512, 1)
        self.transform_7 = nn.Conv2d(2048, 512, 3, stride=2)
        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.trans_classifier = nn.Linear(512 * 3, 20)

    def backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x2 = x
        x = self.layer3(x)
        x3 = x
        x = self.layer4(x)
        x4 = x
        return x2, x3, x4

    def gcn(self):
        inp = inp_var
        adj = gen_adj(self.A).detach()
        gcn = self.gc1(inp, adj)
        gcn = self.relu(gcn)
        gcn_b = self.gc2(gcn, adj)
        gcn_a = self.atten(gcn_b)
        return gcn_b, gcn_a

    def fusion(self, x):
        x2, x3, x4 = self.backbone(x)
        x0 = self.classifier(x4)
        feature = x4.view(x4.size(0), x4.size(1), -1).permute(0, 2, 1)
        features = feature.transpose(-2, -1)
        features = self.poolingtrans(features)
        features = features.view(features.size(0), -1)
        x5 = self.transform_7(x4)
        x4 = self.transform_14(x4)
        x3 = self.transform_28(x3)
        h3, h4, h5 = x3.shape[2], x4.shape[2], x5.shape[2]
        h_max = max(h3, h4, h5)
        x3 = F.interpolate(x3, size=(h_max, h_max), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(h_max, h_max), mode='bilinear', align_corners=True)
        x5 = F.interpolate(x5, size=(h_max, h_max), mode='bilinear', align_corners=True)
        mul = x3 + x4 + x5
        # sum = (x3 + x4 + x5)*0.1
        sum = (x3 * x4 * x5) * 0.1
        x3 = x3 + mul
        x4 = x4 + mul
        x5 = x5 + mul
        x5 = x5 * sum
        x3 = F.interpolate(x3, size=(h3, h3), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(h4, h4), mode='bilinear', align_corners=True)
        x5 = F.interpolate(x5, size=(h5, h5), mode='bilinear', align_corners=True)

        feat3 = self.GMP(x3).view(x3.shape[0], -1)
        feat4 = self.GMP(x4).view(x4.shape[0], -1)
        feat5 = self.GMP(x5).view(x5.shape[0], -1)

        feat = torch.cat((feat3, feat4, feat5), dim=1)
        x1 = self.trans_classifier(feat)
        x2 = x0 + x1 #feature map
        gcn_b, gcn_a = self.gcn()
        x3 = 0.3 * torch.matmul(features, gcn_b.transpose(0, 1)) + 0.2 * torch.matmul(features, gcn_a.transpose(0, 1))
        res = x2 + x3
        return res


    def init_weights(self, pretrained=True, cutmix=None):
        if cutmix != '':
            print("backbone params inited by CutMix pretrained model")
            state_dict = torch.load(cutmix)
        elif pretrained:
            print("backbone params inited by Pytorch official model")
            # model_url = model_urls["resnet{}".format(self.depth)]
            # state_dict = model_zoo.load_url(model_url)
            state_dict = torch.load('')

        # model_dict = self.state_dict()
        self.load_state_dict(state_dict, False)