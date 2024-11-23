import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from gkpd import gkpd, gkpd_rank
from gkpd.tensorops import kron

torch.manual_seed(0)
torch.cuda.manual_seed(0)


# 8 bit hydrea implementation
class HyDREA(nn.Module):
    def __init__(self, dim, D, num_classes, device="cuda"):
        super(HyDREA, self).__init__()
        self.num_classes, self.D = num_classes, D
        self.device = device
        self.rp_layer = nn.Linear(dim, D, bias=False).to(device)
        self.class_hvs = torch.zeros(num_classes, D).float().to(device)
        self.class_hvs = nn.parameter.Parameter(data=self.class_hvs)
        self.class_hvs_quantized = None
        self.zero_point = None
        self.scale = None

    def encoding(self, x):
        out = self.rp_layer(x)
        return torch.sign(out)

    def quantize_class_hvs(self):
        qmin = 0
        qmax = 255

        max_val = self.class_hvs.max().item()
        min_val = self.class_hvs.min().item()
        scale = 2 * max(max_val, abs(min_val)) / (qmax - qmin)
        zero_point = 0
        self.zero_point = zero_point
        self.scale = scale
        self.class_hvs_quantized = torch.quantize_per_tensor(
            self.class_hvs, scale, zero_point, torch.qint8
        )

    def forward(self, x):
        out = self.encoding(x)
        self.quantize_class_hvs()
        out = self.similarity(class_hvs=self.class_hvs_quantized, enc_hv=out)
        return out

    def init_class(self, x_train, labels_train):
        out = self.encoding(x_train)
        self.class_hvs.data = (
            torch.zeros(self.num_classes, self.D).float().to(self.device)
        )

        for i in range(x_train.size()[0]):
            self.class_hvs.data[labels_train[i]] += out[i]

    def HD_train_step(self, x_train, y_train, lr=1.0):
        self.quantize_class_hvs()

        shuffle_idx = torch.randperm(x_train.size()[0])
        x_train = x_train[shuffle_idx]
        train_labels = y_train[shuffle_idx]
        enc_hvs = self.encoding(x_train)
        for i in range(enc_hvs.size()[0]):
            sims = self.similarity(
                self.class_hvs_quantized, enc_hvs[i].unsqueeze(dim=0)
            )
            predict = torch.argmax(sims, dim=1)

            if predict != train_labels[i]:
                self.class_hvs.data[predict] -= lr * enc_hvs[i]
                self.class_hvs.data[train_labels[i]] += lr * enc_hvs[i]

            self.quantize_class_hvs()

    def similarity(self, class_hvs, enc_hv):
        class_hvs = torch.dequantize(class_hvs)
        return torch.matmul(enc_hv, class_hvs.t()) / class_hvs.size()[1]


def HD_test(model, x_test, y_test):
    out = model(x_test)
    preds = torch.argmax(out, dim=-1)
    acc = torch.mean((preds == y_test).float())
    return acc


def get_Cosine_margin_HyDREA(model, x_test):
    def cosine_distance(a, b):
        return 1 - torch.cosine_similarity(a[:, None, :], b, dim=-1)

    # Compute mean Hamming distance between class HVS
    class_hvs = torch.dequantize(model.class_hvs_quantized)
    test_enc_hvs = model.encoding(x_test)
    test_Cosine_dist = cosine_distance(test_enc_hvs, class_hvs)

    sorted_test_Cosine_distance, _ = torch.sort(
        test_Cosine_dist, dim=-1, descending=False
    )
    test_enc_hvs_Cosine_margin = (
        (
            sorted_test_Cosine_distance[:, 1:]
            - sorted_test_Cosine_distance[:, 0].unsqueeze(dim=1)
        )
        .mean(dim=1)
        .cpu()
    )
    mean_test_Cosine_margin = torch.mean(test_enc_hvs_Cosine_margin).item()

    return mean_test_Cosine_margin
