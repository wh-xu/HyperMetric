# %%
from hypermetric import test_error

filename = "Kron_test"
testing = False
output_file = "./output{}/{}.xlsx".format("/test_outputs" if testing else "", filename)
log_file = "./log{}/{}.log".format("/test_logs" if testing else "", filename)
# d0 * d0 = dimensions
# d1 * d1 = nFeatures
config = {
    "datasets": ["ucihar"],
    "dimensions": [1024],
    "levels": [2],
    "tables": None,
    "tests_per_experiment": 10,  # for mean and Std.
    "rp_layer_error_rates": [0],
    # "rp_layer_error_rates": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
    "quant_style": [
        "fixed-percentile"
    ],  # can be trainable, fixed, fixed-percentile, or min-quant-loss
    "kron-params": [
        {"rank": 2, "dims": [(16, 28), (32, 28)]},
        {"rank": 8, "dims": [(16, 28), (32, 28)]},
        {"rank": 16, "dims": [(16, 28), (32, 28)]},
        {"rank": 32, "dims": [(16, 28), (32, 28)]},
    ],
    "plot_quant": False,
    "plot_hist": False,
    "HD_only": False,
}

cos_face_loss_params = {
    "mnist": (30, 0.1),
    "isolet": (30, 0.1),
    "cardio3": (30, 0.1),
    "pamap2": (30, 0.1),
    "ucihar": (30, 0.1),
}

test_error.run_test(config, cos_face_loss_params, output_file, log_file)
# %%

testing = False
filename = "HyDREA_test"
output_file = "./output{}/{}.xlsx".format("/test_outputs" if testing else "", filename)

config = {
    "datasets": ["mnist", "pamap2", "ucihar"],
    "dimensions": [512, 1024, 2048],
    "tests_per_experiment": 10,  # for mean and Std.
    "rp_layer_error_rates": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
}

test_error.run_HyDREA_test(config, output_file)


# %%
testing = False
filename = "HyDREA_test"
output_file = "./output{}/{}.xlsx".format("/test_outputs" if testing else "", filename)

config = {
    "datasets": ["mnist", "pamap2", "ucihar"],
    "dimensions": [512, 1024, 2048],
    "tests_per_experiment": 10,  # for mean and Std.
    "rp_layer_error_rates": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
}

test_error.run_HyDREA_test(config, output_file)
