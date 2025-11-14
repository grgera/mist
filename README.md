<p align="center">
<img src="https://github.com/grgera/mist/blob/main/docs/images/mist_logo.png?raw=true" width="50%" alt='mist'>
</p>

**M**utual **I**nformation estimation via **S**upervised **T**raining

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**MIST** is a framework for fully data-driven mutual information (MI) estimation.
It leverages neural networks trained on large meta-datasets of distributions to learn flexible, differentiable MI estimators that generalize across sample sizes, dimensions, and modalities.
The framework supports uncertainty quantification via quantile regression and provides fast, well-calibrated inference suitable for integration into modern ML pipelines.

This repository contains the reference implementation for the preprint *"MIST: Mutual Information via Supervised Training"*.  It includes scripts to reproduce our experiments as well as tools for training and evaluating MIST-style MI estimators.


## Installation


**Install with pip**

```
pip install mist-statinf
```

**Install with conda**

```
conda env create -f environment.yml
conda activate mist-statinf
```

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/grgera/mist) and install it directly from the source code:

```
pip install -e .       
```


## Quickstart: MI on your (X, Y)

If you want to evaluate MI or obtain confidence intervals on your own data using the **MIST** or **MIST-QR** models described in the paper, use the `MISTQuickEstimator`.

### Point MI estimate with a **MIST**
```python
from mist_statinf import MISTQuickEstimator

X, Y = <your data>

mist = MISTQuickEstimator(
    loss="mse",
    checkpoint="checkpoints/mist/weights.ckpt",
)

mi = mist.estimate_point(X, Y)
print("MIST estimate:", mi)
```

### Median MI estimate and quantile-based confidence intervals with **MIST-QR**
```python
from mist_statinf import MISTQuickEstimator 

X, Y = <your data>

mist_qr = MISTQuickEstimator(
    loss="qr",
    checkpoint="checkpoints/mist_qr/weights.ckpt", 
)

mi_median = mist_qr.estimate_point(X, Y)
print("Median MI:", mi_median)

mi_q90 = mist_qr.estimate_point(X, Y, tau=0.90)
print("q90 MI estimate:", mi_q90)

# --- fast quantile-based uncertainty interval ---
interval = mist_qr.estimate_interval_qr(X, Y, lower=0.05, upper=0.95)
print(interval)
```
By default, `MISTQuickEstimator` loads the pretrained models used in the paper from the package’s `checkpoints/` directory, using the architecture defined in `configs/inference/quickstart.yaml`.
You can override both the checkpoint and the architecture if you have your own trained models.

## Evaluating estimators on test sets

If you want to reproduce the experiments from the paper, we recommend evaluating our trained estimators on the provided test sets (**M_test** and **M_test_extended**).

Since the test sets take a considerable amount of storage space, we publish them separately on Zenodo.  
Before running inference, download the desired subset (either `M_test` or `M_test_extended`).  
Below we show an example using **M_test**, as it is significantly lighter.

```bash
mist-statinf get-data --preset m_test_imd --dir data/test_imd_data
mist-statinf get-data --preset m_test_oomd --dir data/test_oomd_data
```

The simplest way to run inference on these datasets is:
```bash
mist-statinf infer configs/inference/mist_inference.yaml "checkpoints/mist/" 
```

> **_NOTE:_**  The file `mist_inference.yaml` allows you to configure the evaluation mode
(bootstrap or QCQR calibration), select the specific test subset, and specify
which quantiles to compute.

Below we show the results we obtained on **M_test**:
<p align="center">
<img src="https://github.com/grgera/mist/blob/main/docs/images/m_test_results.jpg?raw=true" width="95%" alt='mist'>
</p>


## Train your own MIST Estimators

If you want to reproduce the full training pipeline from the paper — possibly with your own modifications — we recommend following the workflow below.


### 1. Data Generation
```bash
mist-statinf generate configs/data_generation/train.yaml # the same for test and val
```
The generated datasets and their corresponding configuration files will appear under
`data/train_data` and etc.

### 2. Train a MIST Model
```bash
mist-statinf train configs/train/mist_train.yaml
```
Inside the training config you can switch between MSE training and QCQR training.
After training, logs, configs, and the saved model checkpoint will be stored under: `logs/mist_train/run_YYYYmmdd-HHMMSS`

### 3. Running Baselines
```bash
mist-statinf baselines configs/inference/baselines.yaml
```
Baseline results, logs, and configs will be saved to: `logs/bmi_baselines`.

### 3. Test Stage
```bash
mist-statinf infer configs/inference/mist_inference.yaml "logs/mist_train/run_YYYYmmdd-HHMMSS"
```
This will produce CSV predictions and a JSON summary in the same run directory: `logs/mist_train/run_YYYYmmdd-HHMMSS`.

### 4*. (Optional) Hyperparameter Search
```bash
mist-statinf tune logs/mist_train/run_YYYYmmdd-HHMMSS --model-type MSE --n-trials 30
```
This performs a parameter search (via Optuna) starting from a given training run.

## Citation

If you use **MIST** or **MIST-QR** in your work, please cite:

```bibtex
@article{mist2025,
  title   = {MIST: Mutual Information via Supervised Training},
  author  = {},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025}
}
```

## Authors ##

[German Gritsai](https://github.com/), [Megan Richards](https://github.com/), [Maxime Meloux](https://github.com/), [Kyunghyun Cho](https://github.com/), [Maxime Peyrard](https://github.com/).
