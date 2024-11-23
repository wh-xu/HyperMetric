# %%
import pandas as pd
from . import main_hypermetric
import numpy as np
import scipy.stats as st


def run_test(config, cos_face_loss_params, output_file, log_file):
    D = config["dimensions"]
    levels = config["levels"]
    tests_per_experiment = config["tests_per_experiment"]
    tables = config["tables"]
    if tables == None:
        tables = ["table1"]
    datasets = config["datasets"]
    quant_style = config["quant_style"]
    plot_quant = config["plot_quant"]
    plot_hist = config["plot_hist"]
    HD_only = config["HD_only"]
    rp_layer_error_rates = config["rp_layer_error_rates"]
    kron_params = config["kron-params"]

    large_df = pd.DataFrame(
        columns=[
            "dataset",
            "kron_rank",
            "kron_dims",
            "quant_style",
            "level",
            "dimension",
            "table",
            "rp_layer_error_rate",
            "Accuracy",
            "Margin",
        ]
    )
    df = pd.DataFrame(
        columns=[
            "dataset",
            "kron_rank",
            "kron_dims",
            "quant_style",
            "level",
            "dimension",
            "table",
            "rp_layer_error_rate",
            "Average Accuracy",
            "Average Margin",
            "95% Accuracy Confidence Interval",
            "95% Margin Confidence Interval",
            "Std Accuracy",
            "Std Margin",
        ]
    )

    for dataset in datasets:
        s = cos_face_loss_params[dataset][0]
        m = cos_face_loss_params[dataset][1]
        res = {}
        for q in quant_style:
            for level in levels:
                for dim in D:
                    for kron_param in kron_params:
                        acc, marg, MODEL_PATH, logger = main_hypermetric.test(
                            dataset,
                            kron_param,
                            level,
                            dim,
                            s,
                            m,
                            log_file,
                            q,
                            plot_quant,
                            plot_hist,
                            HD_only,
                        )

                        kron_d = None
                        kron_r = None
                        if kron_param is not None:
                            kron_d = kron_param["dims"]
                            kron_r = kron_param["rank"]
                        for i, table in enumerate(tables):
                            for rp_layer_error_rate in rp_layer_error_rates:
                                accuracies = []
                                margins = []
                                for j in range(tests_per_experiment):
                                    acc, marg = main_hypermetric.test_error(
                                        MODEL_PATH,
                                        kron_param,
                                        level,
                                        dim,
                                        dataset,
                                        table,
                                        rp_layer_error_rate,
                                        logger,
                                        q,
                                    )
                                    large_df.loc[len(large_df.index)] = [
                                        dataset,
                                        kron_r,
                                        kron_d,
                                        q,
                                        level,
                                        dim,
                                        table,
                                        rp_layer_error_rate,
                                        acc,
                                        marg,
                                    ]
                                    accuracies.append(acc.item())
                                    margins.append(marg)

                                print(accuracies)
                                print(margins)
                                confid_acc = st.t.interval(
                                    0.95,
                                    len(accuracies) - 1,
                                    loc=np.mean(accuracies),
                                    scale=st.sem(accuracies),
                                )
                                confid_marg = st.t.interval(
                                    0.95,
                                    len(margins) - 1,
                                    loc=np.mean(margins),
                                    scale=st.sem(margins),
                                )

                                df.loc[len(df.index)] = [
                                    dataset,
                                    kron_r,
                                    kron_d,
                                    q,
                                    level,
                                    dim,
                                    table,
                                    rp_layer_error_rate,
                                    np.mean(accuracies),
                                    np.mean(margins),
                                    confid_acc,
                                    confid_marg,
                                    np.var(accuracies),
                                    np.var(margins),
                                ]
                                print(
                                    f"Dataset: {dataset}, Quant Style: {q}, Level: {level}, Dimension: {dim}, Table: {table}, Rp_layer_error_rate: {rp_layer_error_rate}, Accuracy: {np.mean(accuracies)}, Margin: {np.mean(margins)}, 95% Accuracy Confidence Interval: {confid_acc}, 95% Margin Confidence Interval: {confid_marg}, Std Accuracy: {np.var(accuracies)}, Std Margin: {np.var(margins)}"
                                )
                                df.to_excel(output_file, index=False, engine="openpyxl")
                                print(f"DataFrame has been written to {output_file}")

                                file_name = output_file.replace(".xlsx", "_large.xlsx")
                                large_df.to_excel(
                                    file_name, index=False, engine="openpyxl"
                                )
                                print(
                                    f"Large DataFrame has been written to {file_name}"
                                )


def run_HyDREA_test(config, output_file):
    datasets = config["datasets"]
    rp_layer_error_rates = config["rp_layer_error_rates"]
    dimensions = config["dimensions"]
    tests_per_experiment = config["tests_per_experiment"]
    df = pd.DataFrame(
        columns=[
            "dataset",
            "dimension",
            "rp_layer_error_rate",
            "Average Accuracy",
            "Average Margin",
            "95% Accuracy Confidence Interval",
            "95% Margin Confidence Interval",
            "Std Accuracy",
            "Std Margin",
        ]
    )

    for dataset in datasets:
        for dim in dimensions:
            acc, marg, MODEL_PATH = main_hypermetric.test_HyDREA(dataset, dim)
            for rp_error_rate in rp_layer_error_rates:
                accuracies = []
                margins = []
                for i in range(tests_per_experiment):
                    acc, marg = main_hypermetric.test_error_HyDREA(
                        MODEL_PATH, dataset, dim, rp_error_rate
                    )
                    accuracies.append(acc.item())
                    margins.append(marg)
                confid_acc = st.t.interval(
                    0.95,
                    len(accuracies) - 1,
                    loc=np.mean(accuracies),
                    scale=st.sem(accuracies),
                )
                confid_marg = st.t.interval(
                    0.95, len(margins) - 1, loc=np.mean(margins), scale=st.sem(margins)
                )
                df.loc[len(df.index)] = [
                    dataset,
                    dim,
                    rp_error_rate,
                    np.mean(accuracies),
                    np.mean(margins),
                    confid_acc,
                    confid_marg,
                    np.var(accuracies),
                    np.var(margins),
                ]
                print(
                    f"Dataset: {dataset}, Dimension: {dim}, Rp_layer_error_rate: {rp_error_rate}, Accuracy: {np.mean(accuracies)}, Margin: {np.mean(margins)}, 95% Accuracy Confidence Interval: {confid_acc}, 95% Margin Confidence Interval: {confid_marg}, Std Accuracy: {np.var(accuracies)}, Std Margin: {np.var(margins)}"
                )
                df.to_excel(output_file, index=False, engine="openpyxl")
                print(f"DataFrame has been written to {output_file}")
