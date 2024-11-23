# %%
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from .RRAM import RRAM

from gkpd import gkpd, gkpd_rank
from gkpd.tensorops import kron

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def plot_hist(model, name, logger):
    logger.info("level count at " + name)
    plt.title(name)
    uniques, counts = np.unique(model.get_rounded_class_hvs(), return_counts=True)
    logger.info("counts: " + ",".join(str(count) for count in counts.tolist()))
    logger.info("unique: " + ",".join(str(unique) for unique in uniques.tolist()))
    plt.bar(uniques, counts)
    plt.show()


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.outer(w1, w2).clamp(min=eps)


class TripletMarginLoss(nn.Module):
    def __init__(self, loss_fn):
        super(TripletMarginLoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, emb, labels, W=None):
        return self.loss_fn(emb, labels)


class CosFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.40):
        super(CosFaceLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, emb, labels, W):
        logits = cosine_sim(emb, W)

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = self.s * (logits - one_hot * self.m)
        loss = F.cross_entropy(logits, labels)
        return loss


class MetricKDLoss(nn.Module):
    def __init__(self, alpha=0.9, T=4):
        super(MetricKDLoss, self).__init__()
        self.alpha = alpha
        self.T = T

    def forward(self, emb, labels, W, teacher_emb=None):
        if teacher_emb is not None:
            KD_metric_loss = nn.KLDivLoss()(
                F.log_softmax(emb / self.T, dim=1),
                F.softmax(teacher_emb / self.T, dim=1),
            ) * (self.alpha * self.T * self.T) + CosFaceLoss()(emb, labels, W) * (
                1.0 - self.alpha
            )
        else:
            KD_metric_loss = self.loss_metric(emb, labels)
        return KD_metric_loss


def binarize_hard(x):
    return torch.where(x > 0, 1.0, -1.0)


def binarize_soft(x):
    return torch.tanh(x)


def weight_binarize(W):
    W = torch.where(W < -1, -1, W)
    W = torch.where(W > 1, 1, W)
    mask1 = (W >= -1) & (W < 0)
    W[mask1] = 2 * W[mask1] + W[mask1] * W[mask1]
    mask2 = (W >= 0) & (W < 1)
    W[mask2] = 2 * W[mask2] - W[mask2] * W[mask2]
    W = W.float()
    return W


from collections import namedtuple

QTensor = namedtuple("QTensor", ["tensor", "scale", "zero_point"])


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next
    qmin = 0.0
    qmax = 2.0**num_bits - 1.0

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.0
    qmax = 2.0**num_bits - 1.0

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


def plot_quant(model, label):
    x = torch.linspace(
        model.quant_layer.weights[0].item() - 5,
        model.quant_layer.weights[len(model.quant_layer.weights) - 1].item() + 5,
        1000,
    )

    y1 = model.quant_layer(x, soft=False)
    y2 = model.quant_layer(x, soft=True).detach()

    plt.plot(x, y1, c="red", label="Hard Quantization")
    plt.plot(x, y2, c="blue", label="Soft Quantization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Quantize function over learning operations. Label:{}".format(label))
    plt.legend()
    plt.show()
    print("Weights: ", model.quant_layer.weights)


class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8, min_val=None, max_val=None):
        x = quantize_tensor(x, num_bits=num_bits, min_val=min_val, max_val=max_val)
        x = dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class quantize_layer(nn.Module):
    def __init__(self, size_in, train_x, levels, quant_style) -> None:
        super().__init__()
        self.size = size_in
        self.levels = levels

        # weights represent boundry cutoffs
        # initialize weights as even percentiles of training data
        if quant_style == "trainable":
            q = (100 / self.levels) * np.arange(1, self.levels)
            weights = torch.tensor(
                np.percentile(train_x.cpu().flatten().detach(), q)
            ).float()
            self.weights = nn.Parameter(weights)
        elif quant_style == "fixed-percentile":
            q = (100 / self.levels) * np.arange(1, self.levels)
            self.weights = torch.tensor(
                np.percentile(train_x.cpu().flatten().detach(), q)
            ).float()
        elif quant_style == "fixed":
            self.weights = torch.linspace(
                train_x.min().item(), train_x.max().item(), self.levels + 1
            )[1:-1]
        elif quant_style == "min_quant_loss":
            # TODO IMPLEMENT
            pass
        else:
            pass

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


# Model definition for teacher model
class NB_HD_RP(nn.Module):
    def __init__(
        self,
        dim,
        D,
        num_classes,
        x_train,
        levels=2,
        quant_style="trainable",
        device="cpu",
        kargs=None,
    ):
        super().__init__()
        self.num_classes, self.D = num_classes, D
        self.levels = levels
        self.device = device

        self.rp_layer = nn.Linear(dim, D, bias=False).to(device)
        self.class_hvs = torch.zeros(num_classes, D).float().to(device)
        self.class_hvs = nn.parameter.Parameter(data=self.class_hvs)
        self.class_hvs_nb = torch.zeros(num_classes, D).float().to(device)

        self.quant_layer = quantize_layer(
            D, self.rp_layer(x_train), levels, quant_style
        ).to(device)

    def get_rounded_class_hvs(self):
        rounded_class_hvs = torch.round(self.class_hvs.clone().detach())
        if self.levels == 2:
            rounded_class_hvs = torch.where(rounded_class_hvs >= 0, 1.0, -1.0)
            return rounded_class_hvs
        high = (self.levels - 1) // 2
        low = -(self.levels // 2)
        rounded_class_hvs = torch.where(
            rounded_class_hvs >= high, high, rounded_class_hvs
        )
        rounded_class_hvs = torch.where(
            rounded_class_hvs <= low, low, rounded_class_hvs
        )
        return rounded_class_hvs

    def encoding(self, x, soft=False):
        out = self.rp_layer(x)
        return self.quant_layer(out, soft=soft)

    # Forward Function
    def forward(self, x, embedding=True):
        if embedding:
            out = self.encoding(x, soft=True)
        else:
            out = self.encoding(x)
            out = self.similarity(class_hvs=self.get_rounded_class_hvs(), enc_hv=out)
        return out

    def init_class(self, x_train, labels_train):
        out = self.encoding(x_train)

        self.class_hvs_nb = (
            torch.zeros(self.num_classes, self.D).float().to(self.device)
        )
        self.class_hvs.data = (
            torch.zeros(self.num_classes, self.D).float().to(self.device)
        )

        for i in range(x_train.size()[0]):
            self.class_hvs.data[labels_train[i]] += out[i]

        # self.class_hvs.data = self.quant_layer(self.class_hvs)

        self.class_hvs = nn.parameter.Parameter(data=self.quant_layer(self.class_hvs))

    def HD_train_step(self, x_train, y_train, lr=1.0):
        shuffle_idx = torch.randperm(x_train.size()[0])
        x_train = x_train[shuffle_idx]
        train_labels = y_train[shuffle_idx]
        enc_hvs = self.encoding(x_train)
        for i in range(enc_hvs.size()[0]):
            sims = self.similarity(self.class_hvs, enc_hvs[i].unsqueeze(dim=0))
            predict = torch.argmax(sims, dim=1)

            if predict != train_labels[i]:
                self.class_hvs_nb[predict] -= lr * enc_hvs[i]
                self.class_hvs_nb[train_labels[i]] += lr * enc_hvs[i]

            self.class_hvs.data = self.quant_layer(self.class_hvs_nb)

    def similarity(self, class_hvs, enc_hv):
        return torch.matmul(enc_hv, class_hvs.t()) / class_hvs.size()[1]


class HD_Kron_new(nn.Module):
    def __init__(
        self,
        W_init,
        Kron_dims,
        rank,
        num_classes,
        x_train,
        levels,
        quant_style,
        device="cpu",
    ):
        super().__init__()

        self.factor = len(Kron_dims)
        self.Kron_dims = Kron_dims

        self.factor_a, self.factor_b = gkpd_rank(
            W_init, self.Kron_dims[0], self.Kron_dims[1], rank
        )

        self.factor_a = torch.nn.Parameter(data=self.factor_a.to(device))
        self.factor_b = torch.nn.Parameter(data=self.factor_b.to(device))

        self.D, self.dim = list(np.prod(self.Kron_dims, axis=0))
        self.num_classes = num_classes
        self.levels = levels

        out = torch.matmul(x_train, self.get_kron_W().t())
        self.quant_layer = quantize_layer(self.D, out, levels, quant_style).to(device)
        self.device = device

        self.class_hvs = torch.zeros(self.num_classes, self.D).float().to(device)
        self.class_hvs = nn.parameter.Parameter(data=self.class_hvs)
        self.class_hvs_nb = torch.zeros(self.num_classes, self.D).float().to(device)

    def get_kron_W(self):
        return kron(self.factor_a, self.factor_b)

    def get_rounded_class_hvs(self):
        rounded_class_hvs = torch.round(self.class_hvs.clone().detach())
        if self.levels == 2:
            rounded_class_hvs = torch.where(rounded_class_hvs >= 0, 1.0, -1.0)
            return rounded_class_hvs
        high = (self.levels - 1) // 2
        low = -(self.levels // 2)
        rounded_class_hvs = torch.where(
            rounded_class_hvs >= high, high, rounded_class_hvs
        )
        rounded_class_hvs = torch.where(
            rounded_class_hvs <= low, low, rounded_class_hvs
        )
        return rounded_class_hvs

    def encoding(self, x, soft=False):
        out = torch.matmul(x, self.get_kron_W().t())
        return self.quant_layer(out, soft=soft)

    # Forward Function
    def forward(self, x, embedding=True):
        if embedding:
            out = self.encoding(x, soft=True)
        else:
            out = self.encoding(x)
            out = self.similarity(class_hvs=self.get_rounded_class_hvs(), enc_hv=out)
        return out

    def init_class(self, x_train, labels_train):
        out = self.encoding(x_train)

        self.class_hvs_nb = (
            torch.zeros(self.num_classes, self.D).float().to(self.device)
        )
        self.class_hvs.data = (
            torch.zeros(self.num_classes, self.D).float().to(self.device)
        )

        for i in range(x_train.size()[0]):
            self.class_hvs.data[labels_train[i]] += out[i]

        self.class_hvs = nn.parameter.Parameter(data=self.quant_layer(self.class_hvs))

    def HD_train_step(self, x_train, y_train, lr=1.0):
        shuffle_idx = torch.randperm(x_train.size()[0])
        x_train = x_train[shuffle_idx]
        train_labels = y_train[shuffle_idx]
        enc_hvs = self.encoding(x_train)
        for i in range(enc_hvs.size()[0]):
            sims = self.similarity(self.class_hvs, enc_hvs[i].unsqueeze(dim=0))
            predict = torch.argmax(sims, dim=1)

            if predict != train_labels[i]:
                self.class_hvs_nb[predict] -= lr * enc_hvs[i]
                self.class_hvs_nb[train_labels[i]] += lr * enc_hvs[i]

            self.class_hvs.data = self.quant_layer(self.class_hvs_nb)

    def similarity(self, class_hvs, enc_hv):
        return torch.matmul(enc_hv, class_hvs.t()) / class_hvs.size()[1]


def HD_test(model, x_test, y_test):
    out = model(x_test, embedding=False)
    preds = torch.argmax(out, dim=-1)
    acc = torch.mean((preds == y_test).float())
    return acc


def get_Hamming_margin(model, x_test, y_test=None):
    def Hamming_distance(a, b):
        D = a.size()[1]
        return (D - a @ b.T) / 2

    # Compute mean Hamming distance between class HVS
    class_hvs = binarize_hard(model.class_hvs.data)
    class_Hamming_distance = Hamming_distance(class_hvs, class_hvs)
    mean_class_Hamming_distance = torch.mean(class_Hamming_distance).item()

    # Compute test samples' Hamming distance
    test_enc_hvs = binarize_hard(model(x_test, True))
    test_Hamming_dist = Hamming_distance(test_enc_hvs, class_hvs)

    sorted_test_Hamming_distance, _ = torch.sort(
        test_Hamming_dist, dim=-1, descending=False
    )
    test_enc_hvs_Hamming_margin = (
        (
            sorted_test_Hamming_distance[:, 1:]
            - sorted_test_Hamming_distance[:, 0].unsqueeze(dim=1)
        )
        .mean(dim=1)
        .cuda()
    )
    mean_test_Hamming_margin = torch.mean(test_enc_hvs_Hamming_margin).item()

    res_dict = {
        "avg_class_Hamming_dist": mean_class_Hamming_distance,
        "avg_test_Hamming_margin": mean_test_Hamming_margin,
    }
    return mean_test_Hamming_margin


def get_Cosine_margin(model, x_test, soft=False):
    def cosine_distance(a, b):
        return 1 - torch.cosine_similarity(a[:, None, :], b, dim=-1)

    # Compute mean Hamming distance between class HVS
    class_hvs = model.class_hvs.data if soft else model.get_rounded_class_hvs()
    class_Cosine_distance = cosine_distance(class_hvs, class_hvs)
    mean_class_Cosine_distance = torch.mean(class_Cosine_distance).item()

    # Compute test samples' Hamming distance
    test_enc_hvs = model.encoding(x_test, soft=soft)
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
    return mean_test_Cosine_margin




def test_RRAM_HD(model, x_test, y_test, kargs, trial_num=0):
    class_hvs = binarize_hard(model.class_hvs)
    test_hvs = binarize_hard(model(x_test, embedding=True))

    preds = torch.argmax(test_hvs @ class_hvs.T, dim=-1)
    fp_acc = torch.mean((preds == y_test).float()).item()

    unipolar_class_hvs = torch.where(class_hvs > 0, 1, 0).cpu().numpy()
    unipolar_test_hvs = torch.where(test_hvs > 0, 1, 0).cpu().numpy()

    rram_chip = RRAM(S_ou=kargs["S_ou"], R=kargs["R"], R_deviation=kargs["R_deviation"])

    res_dict = {
        "S_ou": kargs["S_ou"],
        "R": kargs["R"],
        "R_deviation": kargs["R_deviation"],
        "fp_acc": fp_acc,
        "test_acc": [],
    }

    for i in range(trial_num):
        rram_chip.rram_write_binary(unipolar_class_hvs)

        preds, Hamming_sim_cim = rram_chip.rram_hd_am(
            unipolar_test_hvs, collect_stats=True
        )

        acc = torch.mean((torch.tensor(preds).cuda() == y_test).float()).item()
        res_dict["test_acc"].append(acc)

    return res_dict


def metric_train(
    model,
    epochs,
    train_loader,
    x_test,
    y_test,
    kargs_hd_rram_test,
    loss_func,
    optimizer,
    logger,
    device,
):

    model.train()

    best_acc = -1e6
    for epoch_i in range(1, epochs + 1):
        logger.info(
            "\n=====================================\nMetric learning epoch: {}".format(
                epoch_i
            )
        )
        final_loss = 0
        iter_count = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings = model(data.reshape(data.size()[0], -1))
            loss = loss_func(embeddings, labels, model.class_hvs)

            loss.backward()
            optimizer.step()

            final_loss += loss.item()
            iter_count += 1
        logger.info("AVG Loss: {}".format(final_loss / iter_count))

        # res_Hamming = get_Hamming_margin(model, x_test)
        # res_Hamming = get_Cosine_margin(model, x_test, soft=True)
        # logger.info("Soft Hamming margin: {}".format(res_Hamming))
        res_Hamming = get_Cosine_margin(model, x_test, soft=False)
        logger.info("Hard Hamming margin: {}".format(res_Hamming))

        acc = HD_test(model, x_test, y_test).item()
        logger.info("Acc: {}".format(acc))

        if acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = acc

    return best_model, best_acc


def metric_kd_train(
    model,
    teacher_model,
    epochs,
    train_loader,
    x_test,
    y_test,
    kargs_hd_rram_test,
    loss_func,
    optimizer,
    logger,
    device,
):

    model.train()

    best_acc = -1e6
    for epoch_i in range(1, epochs + 1):
        logger.info(
            "\n=====================================\nMetric learning epoch: {}".format(
                epoch_i
            )
        )
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            inp = data.reshape(data.size()[0], -1)
            embeddings = model(inp)
            with torch.no_grad():
                teacher_emb = teacher_model(inp)
            loss = loss_func(embeddings, labels, model.class_hvs, teacher_emb)

            loss.backward()
            optimizer.step()

        # res_Hamming = get_Hamming_margin(model, x_test)
        res_Hamming = get_Cosine_margin(model, x_test, soft=True)
        logger.info("Soft Hamming margin: {}".format(res_Hamming))
        res_Hamming = get_Cosine_margin(model, x_test, soft=False)
        logger.info("Hard Hamming margin: {}".format(res_Hamming))

        res_rram = test_RRAM_HD(model, x_test, y_test, kargs_hd_rram_test)
        logger.info(res_rram)

        if res_rram["fp_acc"] > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = res_rram["fp_acc"]

    return best_model, best_acc


def hd_train(
    model, epochs, x_train, y_train, x_test, y_test, kargs_hd_rram_test, logger, device
):

    model.train()
    best_acc = -1e6
    for epoch_i in range(1, epochs + 1):
        logger.info(
            "\n=====================================\nHD learning epoch: {}".format(
                epoch_i
            )
        )

        model.HD_train_step(x_train, y_train)

        # res_Hamming = get_Hamming_margin(model, x_test)
        # res_Hamming = get_Cosine_margin(model, x_test, soft=True)
        # logger.info("Soft Hamming margin: {}".format(res_Hamming))
        res_Hamming = get_Cosine_margin(model, x_test, soft=False)
        logger.info("Hard Hamming margin: {}".format(res_Hamming))

        acc = HD_test(model, x_test, y_test).item()
        logger.info("Acc: {}".format(acc))

        if acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = acc

        # logger.info("Roudned Class_hvs: {}".format(model.get_rounded_class_hvs()))

    # Restore the best model
    model = best_model
    return model, best_acc
