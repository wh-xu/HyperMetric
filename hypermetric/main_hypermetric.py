# %%
import os, sys
import argparse
import logging
from struct import *
import numpy as np

from . import models, dataset_utils, error_rates, HyDREA

from gkpd import gkpd_rank
from gkpd.tensorops import kron

import torch
import torch.nn as nn
import torch.optim as optim

# %%
logging.getLogger("matplotlib.font_manager").disabled = True


def test(
    dataset,
    kron_params,
    levels,
    dimension,
    s,
    m,
    log_file,
    quant_style,
    plot_quant,
    plot_hist,
    HD_only,
    model_load_path=None,
):
    args = argparse.Namespace(
        dataset=dataset,
        model_path=None,
        batch_size=256,
        D=dimension,
        epoch_HDC=10,
        epoch_metric=10,
        kron_dims=[],
        kron_rank=4,
        epoch_kd=0,
        lr=0.001,
        S_ou=8,
        log_path="./log",
    )
    device = "cpu"

    batch_size = 256
    D = dimension
    num_HD_epochs = 10
    num_metric_epochs = 10
    lr_metric = 0.001

    num_kd_epochs = args.epoch_kd
    lr_kd = args.lr
    logging.basicConfig(
        filename=log_file, format="%(asctime)s %(message)s", filemode="w"
    )
    logger = logging.getLogger()
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logger.setLevel(logging.DEBUG)
    kargs_hd_rram_test = {
        "S_ou": args.S_ou,
        "R": [2500, 16000],
        "R_deviation": [0.18, 0.45],
    }

    all_results = {
        "train_type": [],
        "epoch": [],
        "avg_class_Hamming_margin": [],
        "avg_test_Hamming_margin": [],
        "fp_acc": [],
        "reram_acc": [],
        "ckpt_filename": [],
    }

    def update_result_dict(
        train_type, epoch_i, dict_Hamming_margin, reram_test_dict, ckpt_filename
    ):
        all_results["train_type"].append(train_type)
        all_results["epoch"].append(epoch_i)
        all_results["avg_class_Hamming_margin"].append(
            dict_Hamming_margin["avg_class_Hamming_dist"]
        )
        all_results["avg_test_Hamming_margin"].append(
            dict_Hamming_margin["avg_test_Hamming_margin"]
        )
        all_results["fp_acc"].append(reram_test_dict["fp_acc"])
        all_results["reram_acc"].append(reram_test_dict["test_acc"])
        all_results["ckpt_filename"].append(ckpt_filename)

    # Load datasets
    nFeatures, nClasses, x_train, y_train, x_test, y_test, train_loader, test_loader = (
        dataset_utils.load_dataset(dataset, batch_size, device)
    )
    logger.info(
        "Loaded dataset: {} with {} features, {} classes, train size={}, test size={}".format(
            dataset, nFeatures, nClasses, len(x_train), len(x_test)
        )
    )

    if model_load_path is None:
        model_hd = models.NB_HD_RP(
            dim=nFeatures,
            D=D,
            num_classes=nClasses,
            x_train=x_train,
            levels=levels,
            quant_style=quant_style,
            device=device,
        )
        model_hd.init_class(x_train, y_train)

        ### pytorch-metric-learning stuff ###

        loss_func = models.CosFaceLoss(s=s, m=m)

        optimizer = optim.Adam(model_hd.parameters(), lr=lr_metric)
        # optimizer = optim.SGD(model_hd.parameters(), lr=metric_lr)
        model_hd.rp_layer.weight.requires_grad = True
        model_hd.class_hvs.requires_grad = False

        if plot_quant:
            models.plot_quant(model_hd, "After Initialization")

        if plot_hist:
            models.plot_hist(model_hd, "After Initialization", logger)

        if not HD_only:
            models.metric_train(
                model_hd,
                num_metric_epochs,
                train_loader,
                x_test,
                y_test,
                kargs_hd_rram_test,
                loss_func,
                optimizer,
                logger,
                device,
            )

        if plot_hist:
            models.plot_hist(model_hd, "After Metric Train Round 1", logger)

        if plot_quant:
            models.plot_quant(model_hd, "After Metric Train Round 1")

        # 2. HD Training

        best_model, best_acc = models.hd_train(
            model_hd,
            num_HD_epochs,
            x_train,
            y_train,
            x_test,
            y_test,
            kargs_hd_rram_test,
            logger,
            device,
        )

        if plot_hist:
            models.plot_hist(model_hd, "After HD training", logger)

        if plot_quant:
            models.plot_quant(model_hd, "After HD training")

        # 3. Metric Training for Class HVS
        loss_func = models.CosFaceLoss()
        optimizer = optim.Adam(model_hd.parameters(), lr=lr_metric)
        model_hd.rp_layer.weight.requires_grad = True
        model_hd.class_hvs.requires_grad = True

        if not HD_only:
            best_model, best_acc = models.metric_train(
                model_hd,
                num_metric_epochs,
                train_loader,
                x_test,
                y_test,
                kargs_hd_rram_test,
                loss_func,
                optimizer,
                logger,
                device,
            )

        if plot_hist:
            models.plot_hist(model_hd, "After Metric Train Final Round", logger)

        if plot_quant:
            models.plot_quant(model_hd, "After Metric Train Final Round")

        # update_result_dict('hdc', epoch_i, res_Hamming, res_rram, MODEL_FILENAME)

        MODEL_FILENAME = "model_hd_metric_acc_{:.3f}_levels_{}_dim_{}_dataset_{}_quant_{}.ckpt".format(
            best_acc, levels, D, dataset, quant_style
        )
        MODEL_PATH = os.path.join("./models", MODEL_FILENAME)
        torch.save(best_model.state_dict(), MODEL_PATH)
        logger.info(
            "Saved best model with {:.3f} acc. to {}".format(best_acc, MODEL_PATH)
        )

        final_margin = models.get_Cosine_margin(model_hd, x_test, soft=False)
    else:
        logger.info("Loading model at {}".format(model_load_path))
        model_hd = models.NB_HD_RP(
            dim=nFeatures,
            D=D,
            num_classes=nClasses,
            x_train=x_train,
            levels=levels,
            device=device,
            quant_style=quant_style,
        )
        model_hd.load_state_dict(torch.load(model_load_path))

    if kron_params:
        kron_rank = kron_params["rank"]
        kron_dims = kron_params["dims"]

        # Initialize Kron's weight using SVD
        W_pretrain = model_hd.rp_layer.weight.data.detach().to(device)
        model_kron = models.HD_Kron_new(
            W_init=W_pretrain,
            Kron_dims=kron_dims,
            rank=kron_rank,
            num_classes=nClasses,
            x_train=x_train,
            levels=levels,
            device=device,
            quant_style=quant_style,
        )
        model_kron.init_class(x_train=x_train, labels_train=y_train)

        # Test reconstruction error
        W_hat = model_kron.get_kron_W()

        def fake_kron(W):
            factor_a, factor_b = gkpd_rank(W, kron_dims[0], kron_dims[1], kron_rank)
            return kron(factor_a, factor_b)

        torch.save(W_pretrain, "pretrain_{}.pt".format(kron_rank))

        err = torch.abs((W_hat - W_pretrain) / W_pretrain).mean()
        logger.info(
            "For {} Levels, {} Dimensions, and {} kron_params: Reconstruction error = {}".format(
                levels, D, kron_params, err
            )
        )

        fake_err = torch.abs((fake_kron(W_hat) - W_pretrain) / W_pretrain).mean()
        logger.info("Fake reconstruction error = {}".format(fake_err))

        # 1. Metric training for Kron
        loss_func = models.CosFaceLoss(s=s, m=m)
        optimizer = optim.Adam(model_kron.parameters(), lr=lr_metric)
        model_kron.factor_a.requires_grad = True
        model_kron.factor_b.requires_grad = True
        model_kron.class_hvs.requires_grad = False

        _, _ = models.metric_train(
            model_kron,
            num_metric_epochs,
            train_loader,
            x_test,
            y_test,
            kargs_hd_rram_test,
            loss_func,
            optimizer,
            logger,
            device,
        )

        # 2. HD Training
        # model_kron.init_class(x_train, y_train)
        best_model_hd, best_acc_hd = models.hd_train(
            model_kron,
            num_HD_epochs,
            x_train,
            y_train,
            x_test,
            y_test,
            kargs_hd_rram_test,
            logger,
            device,
        )

        # Restore the best model
        model_kron = best_model_hd

        # 3. Metric Training for Class HVS
        optimizer = optim.Adam(model_kron.parameters(), lr=lr_metric)
        model_kron.factor_a.requires_grad = True
        model_kron.factor_b.requires_grad = True
        model_kron.class_hvs.requires_grad = True

        best_model, best_acc = models.metric_train(
            model_kron,
            num_metric_epochs,
            train_loader,
            x_test,
            y_test,
            kargs_hd_rram_test,
            loss_func,
            optimizer,
            logger,
            device,
        )

        best_model = best_model if num_metric_epochs else best_model_hd
        best_acc = best_acc if num_metric_epochs else best_acc_hd
        final_margin = models.get_Cosine_margin(model_kron, x_test, soft=False)

        if best_model:
            MODEL_FILENAME = "model_kron_acc_{:.3f}.ckpt".format(best_acc)
            MODEL_PATH = os.path.join("./models", MODEL_FILENAME)
            torch.save(best_model.state_dict(), MODEL_PATH)
            logger.info(
                "Saved best model with {:.3f} acc. to {}".format(best_acc, MODEL_PATH)
            )

        return best_acc, final_margin, MODEL_PATH, logger
    else:
        if model_load_path is None:
            return best_acc, final_margin, MODEL_PATH, logger
        else:
            return 0, 0, model_load_path, logger


def apply_error(original, bit_error_rate=0.01):
    qmin = 0
    qmax = 255

    max_val = original.max().item()
    min_val = original.min().item()
    scale = 2 * max(max_val, abs(min_val)) / (qmax - qmin)
    zero_point = 0
    qmin = 0
    qmax = 255

    max_val = original.max().item()
    min_val = original.min().item()
    scale = 2 * max(max_val, abs(min_val)) / (qmax - qmin)
    zero_point = 0

    quantized_weights = torch.quantize_per_tensor(
        original, scale, zero_point, torch.qint8
    )
    quantized_weights = torch.quantize_per_tensor(
        original, scale, zero_point, torch.qint8
    )
    mask = np.random.choice(
        [0, 1],
        size=(quantized_weights.size() + (8,)),
        p=[1.0 - bit_error_rate, bit_error_rate],
    )
    mask_pack = torch.tensor(np.packbits(mask, axis=len(original.size())))
    output = quantized_weights.int_repr() ^ mask_pack.squeeze()
    output = (output - zero_point) * scale
    mask_pack = torch.tensor(np.packbits(mask, axis=len(original.size())))
    output = quantized_weights.int_repr() ^ mask_pack.squeeze()
    output = (output - zero_point) * scale
    return output


def test_error(
    path,
    kron_param,
    level,
    dimension,
    dataset,
    table,
    rp_layer_error_rate,
    logger,
    quant_style,
):
    args = argparse.Namespace(
        dataset=dataset,
        model_path=path,
        batch_size=256,
        D=dimension,
        epoch_HDC=10,
        epoch_metric=10,
        kron_dims=[],
        kron_rank=4,
        epoch_kd=0,
        S_ou=8,
        log_path="./log",
        levels=level,
    )

    logger.info(
        "Beginning Error Test for {} with {} levels, {} dimensions, {}, rp_error_rate, and {} table".format(
            dataset, level, dimension, rp_layer_error_rate, table
        )
    )
    dataset = args.dataset  # 'mnist'
    # cardio3 is not stable
    batch_size = args.batch_size  # 256
    D = args.D  # 1024
    device = "cpu"
    levels = args.levels
    nFeatures, nClasses, x_train, y_train, x_test, y_test, train_loader, test_loader = (
        dataset_utils.load_dataset(dataset, batch_size, device)
    )

    MODEL_FILENAME = args.model_path

    map_table = {
        "table6": error_rates.table6,
        "table5": error_rates.table5,
        "table4": error_rates.table4,
        "table3": error_rates.table3,
        "table2": error_rates.table2,
        "table1": error_rates.table1,
    }

    if kron_param:
        kron_dims = kron_param["dims"]
        kron_rank = kron_param["rank"]
        W_pretrain = torch.zeros((D, nFeatures)).float()

        model_kron = models.HD_Kron_new(
            W_init=W_pretrain,
            Kron_dims=kron_dims,
            rank=kron_rank,
            num_classes=nClasses,
            x_train=x_train,
            levels=levels,
            device=device,
            quant_style=quant_style,
        )
        model_kron.load_state_dict(torch.load(MODEL_FILENAME))

        original = model_kron.get_rounded_class_hvs().clone()
        model_kron.class_hvs = nn.Parameter(
            torch.tensor(
                error_rates.apply_error_rates(
                    model_kron.get_rounded_class_hvs(), map_table[table], levels
                )
            ).float()
        )
        model_kron.factor_a.data = nn.Parameter(
            torch.tensor(
                apply_error(model_kron.factor_a.data, rp_layer_error_rate)
            ).float()
        )
        model_kron.factor_b.data = nn.Parameter(
            torch.tensor(
                apply_error(model_kron.factor_b.data, rp_layer_error_rate)
            ).float()
        )

        class_hv_mae = torch.abs(original - model_kron.get_rounded_class_hvs()).mean()
        logger.info("MAE of class_hv error:{}".format(class_hv_mae.item()))
        after_acc = models.HD_test(model_kron, x_test, y_test)
        after_margin = models.get_Cosine_margin(model_kron, x_test, soft=False)

    else:
        model_hd = models.NB_HD_RP(
            dim=nFeatures,
            D=D,
            num_classes=nClasses,
            x_train=x_train,
            levels=levels,
            device=device,
            quant_style=quant_style,
        )
        model_hd.load_state_dict(torch.load(MODEL_FILENAME))
        original = model_hd.get_rounded_class_hvs().clone()

        model_hd.class_hvs = nn.Parameter(
            torch.tensor(
                error_rates.apply_error_rates(
                    model_hd.get_rounded_class_hvs(), map_table[table], levels
                )
            ).float()
        )
        model_hd.rp_layer.weight = nn.Parameter(
            torch.tensor(
                apply_error(model_hd.rp_layer.weight, rp_layer_error_rate)
            ).float()
        )
        class_hv_mae = torch.abs(original - model_hd.get_rounded_class_hvs()).mean()
        logger.info("MAE of class_hv error:{}".format(class_hv_mae.item()))
        after_acc = models.HD_test(model_hd, x_test, y_test)
        after_margin = models.get_Cosine_margin(model_hd, x_test, soft=False)

    logger.info("Accuracy: {}, Margin: {}".format(after_acc, after_margin))

    return after_acc, after_margin


# %%


def test_HyDREA(dataset, dimension, device="cpu"):
    nFeatures, nClasses, x_train, y_train, x_test, y_test, train_loader, test_loader = (
        dataset_utils.load_dataset(dataset, 128, device)
    )
    model = HyDREA.HyDREA(
        dim=nFeatures, D=dimension, num_classes=nClasses, device=device
    )
    model.train()
    model.init_class(x_train, y_train)
    epochs = 10
    best_acc = -1e6
    MODEL_PATH = "models/HyDREA_{}_{}".format(dataset, dimension)
    acc = HyDREA.HD_test(model, x_test, y_test)
    marg = HyDREA.get_Cosine_margin_HyDREA(model, x_test)
    print(
        "HyDREA HD Training round {}, dimension: {}, Accuracy: {}, Margin {}".format(
            0, dimension, acc, marg
        )
    )
    for e in range(epochs):
        model.HD_train_step(x_train, y_train)
        acc = HyDREA.HD_test(model, x_test, y_test)
        marg = HyDREA.get_Cosine_margin_HyDREA(model, x_test)

        print(
            "HyDREA HD Training round {}, dimension: {}, Accuracy: {}, Margin {}".format(
                e + 1, dimension, acc, marg
            )
        )
        if acc > best_acc:
            torch.save(model.state_dict(), MODEL_PATH)
            best_acc = acc
            best_marg = marg

    return best_acc, best_marg, MODEL_PATH


def test_error_HyDREA(
    MODEL_PATH, dataset, dimension, rp_layer_error_rate, device="cpu"
):
    nFeatures, nClasses, x_train, y_train, x_test, y_test, train_loader, test_loader = (
        dataset_utils.load_dataset(dataset, 128, device)
    )
    model = HyDREA.HyDREA(
        dim=nFeatures, D=dimension, num_classes=nClasses, device=device
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.rp_layer.weight = nn.Parameter(
        torch.tensor(apply_error(model.rp_layer.weight, rp_layer_error_rate)).float()
    )
    acc = HyDREA.HD_test(model, x_test, y_test)
    marg = HyDREA.get_Cosine_margin_HyDREA(model, x_test)
    return acc, marg
