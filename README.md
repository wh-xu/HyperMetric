[![License bsd-3-clause](https://badgen.net/badge/license/MIT/red)](https://github.com/wh-xu/HyperMetric/blob/main/LICENSE)

# HyperMetric
A framework to train lightweight and robust hyperdimensional computing (HDC) models. This repository is the official implementation of [HyperMetric](https://www.computer.org/csdl/proceedings-article/iccd/2023/429100a243).

## Preparing your environment

1. Get yourself a Python environment with version 3.9 to 3.11. The test was done using Python 3.10. Using a `virtualenv` is recommended but not required.

        conda create -n hdc python=3.10
        conda activate hdc

2. HyperMetric requires the following packages to be installed: 

        torch
        pandas
        numpy
        pytorch_metric_learning
        matplotlib
        seaborn

    We recommend using Pytorch on CUDA to accelerate the training process. Please follow the official Pytorch [installation instructions](https://pytorch.org/get-started/locally) to install the GPU version. If use `torch` on CPU, the following command can be ran to installed the dependencies:

        python -m pip install -r requirements.txt

## Usage

Here is an example of how to run the example experiment:

    python run_error_experiments.py

TODO: Add more examples

## Citation and Publication

1. Weihong Xu, Viji Swaminathan, Sumukh Pinge, Sean Fuhrman, and Tajana Rosing. "[HyperMetric: Robust Hyperdimensional Computing on Error-prone Memories using Metric Learning](https://www.computer.org/csdl/proceedings-article/iccd/2023/429100a243)." _IEEE 41st International Conference on Computer Design (ICCD)_, 2023.
2. Weihong Xu, Sean Fuhrman, Keming Fan, Sumukh Pinge, Wei-Chen Chen, and Tajana Rosing. "HyperMetric: Fault-Tolerant Hyperdimensional Computing with Metric Learning for Robust Edge Intelligence" _Under Review_, 2024.

Please cite the following paper if use the idea of this work:
```
@inproceedings{xu2023hypermetric,
  title={HyperMetric: Robust Hyperdimensional Computing on Error-prone Memories using Metric Learning},
  author={Xu, Weihong and Swaminathan, Viji and Pinge, Sumukh and Fuhrman, Sean and Rosing, Tajana},
  booktitle={IEEE 41st International Conference on Computer Design (ICCD)},
  pages={243--246},
  year={2023},
}
```

## License Information

The code is distributed under the MIT License.
