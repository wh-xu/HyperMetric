import torch.nn.functional as F
import torch.nn as nn
import torch

from pytorch_metric_learning import testers

import matplotlib.pyplot as plt
import numpy as np
import sys
from copy import deepcopy

np.random.seed(0)
torch.manual_seed(0)


def plot(self, x, name):
    plt.title(name)
    plt.hist(x)
    plt.show()


torch.set_printoptions(profile="full")


# Cosine margin is the mean cosine distance between test samples and class_hvs
# or between class_hvs and class_hvs
def get_Cosine_margin(model, x_test, y_test=None):
    def cosine_distance(a, b):
        return 1 - torch.cosine_similarity(a[:, np.newaxis, :], b, dim=-1)

    # Compute mean Hamming distance between class HVS
    class_hvs = model.class_hvs.data
    class_Cosine_distance = cosine_distance(class_hvs, class_hvs)
    mean_class_Cosine_distance = torch.mean(class_Cosine_distance).item()
    # print("class hvs:", class_Cosine_distance)

    # Compute test samples' Hamming distance
    test_enc_hvs = model.encoding(x_test, True)
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

    res_dict = {
        "avg_class_Cosine_dist": mean_class_Cosine_distance,
        "avg_test_Cosine_margin": mean_test_Cosine_margin,
    }
    return res_dict


class quantize_layer(nn.Module):
    def __init__(self, size_in, train_x, levels) -> None:
        super().__init__()
        self.size = size_in
        self.levels = levels

        # weights represent boundry cutoffs
        # initialize weights as even percentiles of training data
        q = (100 / self.levels) * np.arange(1, self.levels)
        weights = torch.tensor(
            np.percentile(train_x.cpu().flatten().detach(), q)
        ).float()

        self.weights = nn.Parameter(weights)

    def forward(self, x, soft=False):
        if soft:  # for training

            # set tanh boundries as midpoints of weights
            cutoffs = [float("-inf")]
            for i in range(len(self.weights) - 1):
                mean = ((self.weights[i] + self.weights[i + 1]) / 2).item()
                cutoffs.append(mean)

            # lower/upper boundry are +- infinity
            cutoffs.append(float("inf"))
            res = torch.zeros_like(x)
            offsets = np.arange(int(0 - self.levels / 2) + 0.5, int(self.levels / 2), 1)
            for i in range(len(cutoffs) - 1):
                mask = ((x > cutoffs[i]) & (x < cutoffs[i + 1])).float()
                if self.levels == 2:
                    res += mask * (
                        (torch.tanh((x - self.weights[i]))) + offsets[i] + 0.5
                    )  # rescale for 2 levels
                else:
                    res += mask * (
                        (0.5 * torch.tanh((x - self.weights[i]))) + offsets[i]
                    )

            return res

        else:
            res = torch.zeros_like(x)
            if self.levels == 2:
                # hard quantize based on weights
                for cutoff in self.weights:
                    res += 2 * (x > cutoff).float()  # modifies so 2 levels = [-1, +1]
            else:
                for cutoff in self.weights:
                    res += (x > cutoff).float()

            return res - int(self.levels / 2)


def binarize_hard(x):
    return torch.where(x > 0, 1.0, -1.0)


def binarize_soft(x):
    return torch.tanh(x)


class HDC(nn.Module):
    def __init__(
        self,
        dim,
        D,
        num_classes,
        levels=3,
        enc_type="RP",
        device="cuda",
        similarity_type="cosine",
        kargs=None,
    ):
        super(HDC, self).__init__()
        self.enc_type = enc_type
        self.device = device
        self.D = D
        self.levels = levels
        self.similarity_type = similarity_type

        self.quantize_data = None
        self.quantize_class_hvs = None
        self.class_hvs = torch.zeros(num_classes, D).float().to(device)
        self.class_hvs_nq = torch.zeros(num_classes, D).float().to(device)

        if enc_type in ["RP", "RP-COS"]:
            self.rp_layer = nn.Linear(dim, D).to(device)
        elif enc_type == "ID":
            self.m = m
            self.base_hvs = torch.randint(0, 2, (dim, D)).float().to(device)
            self.level_hvs = self.generate_level_hvs().to(device)

    def init_quantize(self, x_train):
        self.quantize_data = quantize_layer(
            self.D, self.encoding(x_train, quantize=False), levels=self.levels
        )

    def encoding(self, x, soft=False, quantize=True):
        if self.enc_type == "RP":
            out = self.rp_layer(x)
        elif self.enc_type == "RP-COS":
            out = torch.cos(self.rp_layer(x))
        elif self.enc_type == "ID":
            q = (100 / self.m) * np.arange(1, self.m)
            cutoffs = np.percentile(x.flatten().detach(), q).tolist()

            res = torch.zeros_like(x).long()
            for cutoff in cutoffs:
                res += (x > cutoff).long()

            selected_level_hvs = self.level_hvs[res]

            out = torch.logical_xor(selected_level_hvs, self.base_hvs).float()
            out = torch.sum(out, dim=1)
        else:
            raise Exception("Error encoding type: {}".format(self.enc_type))

        return self.quantize_data(out, soft) if quantize else out

    def init_class(self, x_train, labels_train):
        with torch.no_grad():
            out = self.encoding(x_train)

            for i in range(x_train.size()[0]):
                self.class_hvs[labels_train[i]] += out[i]

            # self.class_hvs_nq = self.class_hvs
            self.quantize_class_hvs = quantize_layer(
                self.D, self.class_hvs, levels=self.levels
            )

            self.class_hv_distribution = (
                self.class_hvs.clone().detach()
            )  # for creating plots
            self.class_hvs = nn.parameter.Parameter(
                self.quantize_class_hvs(self.class_hvs)
            )

    def HD_train_step(self, x_train, y_train, lr=1):
        with torch.no_grad():
            shuffle_idx = torch.randperm(x_train.size()[0])
            x_train = x_train[shuffle_idx]
            train_labels = y_train[shuffle_idx]

            enc_hvs = self.encoding(x_train, soft=False)
            for i in range(enc_hvs.size()[0]):
                sims = self.similarity(self.class_hvs, enc_hvs[i].unsqueeze(dim=0))
                predict = torch.argmax(sims, dim=-1)

                if predict != train_labels[i]:
                    self.class_hvs_nq[predict] -= lr * enc_hvs[i]
                    self.class_hvs_nq[train_labels[i]] += lr * enc_hvs[i]

                self.quantize_class_hvs = quantize_layer(
                    self.D, self.class_hvs_nq, levels=self.levels
                )
                self.class_hvs.data = self.quantize_class_hvs(self.class_hvs_nq)

            self.class_hv_distribution = self.class_hvs_nq

    def similarity(self, class_hvs, enc_hv):
        if self.similarity_type == "cosine":
            out = torch.cosine_similarity(enc_hv[:, np.newaxis, :], class_hvs, dim=-1)
        elif self.similarity_type == "matmul":
            out = torch.matmul(enc_hv, class_hvs.t()) / class_hvs.size()[1]
        elif self.similarity_type == "hamming":
            out = torch.sum(enc_hv[:, np.newaxis, :] == class_hvs, dim=-1) / self.D
        else:
            raise Exception("Error similarity type: {}".format(self.similarity_type))
        return out

    def forward(self, x, embedding=True):
        if embedding:
            x.requires_grad = True
            out = self.encoding(x, soft=True)
        else:
            out = self.encoding(x, soft=False)
            out = self.similarity(self.class_hvs, out)
        return out


### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
def metric_train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data.reshape(data.size()[0], -1), embedding=True)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        # if batch_idx % 100 == 0:
        #     print(
        #         "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
        #             epoch, batch_idx, loss, mining_func.num_triplets
        #         )
        #


def quant_train(model, loss_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data.reshape(data.size()[0], -1), embedding=True)
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()
        # if batch_idx % 100 == 0:
        #     print(
        #         "Epoch {} Iteration {}: Loss = {}".format(
        #             epoch, batch_idx, loss
        #         ))


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def metric_test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
