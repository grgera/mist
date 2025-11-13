<p align="center">
<img src="https://raw.githubusercontent.com/grgera/mist/docs/images/mist-logo.png" width="50%" alt='project-monai'>
<!-- <img src="docs/images/mist_logo.png" width="50%" alt="project-monai"> -->
</p>

**M**utual **I**nformation estimation via **S**upervised **T**raining

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**MIST** is a Python package for fully data-driven mutual information (MI) estimation.
It leverages neural networks trained on large meta-datasets of distributions to learn flexible, differentiable MI estimators that generalize across sample sizes, dimensions, and modalities.
The framework supports uncertainty quantification via quantile regression and provides fast, well-calibrated inference suitable for integration into modern ML pipelines.

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
git clone <repo>
cd mist
pip install -e .       
```


## Quickstart: one-line MI on your (X, Y)

If you just want to test on your own data use the QuickStart API.

> **_NOTE:_**  Please note that the first time you call get_encoder_decoder_tokenizer, the models are
being downloaded which might take a minute or two.

```python
import numpy as np
from mist_statinf import MISTQuickEstimator  

# Example data 
X = np.random.randn(1000, 1)
Y = X + 0.3*np.random.randn(1000, 1)

# 1) Point estimate (MSE-trained model)
mist = MISTQuickEstimator(loss="mse", checkpoint="checkpoints/pretrained_mist.pt")
mi = mist.estimate(X, Y)  
print(f"MIST estimate: {mi:.3f}")

# 2) Quantile model (QCQR)
mist_qr = MISTQuickEstimator(loss="qr", checkpoint="checkpoints/pretrained_qcqr.pt", tau=0.5)
res = mist_qr.estimate(X, Y, n_resamples=50, alpha=0.05)
print(f"Median MI: {res['mean']:.3f}  (95% CI [{res['ci_lower']:.3f}, {res['ci_upper']:.3f}])")
```
By default, QuickStart loads an architecture from the package resource `checkpoints/` with
`configs/inference/quickstart.yaml`. You can override it.

## Evaluating proposed estimators on test sets

The simplest way to get started for generation is to use the default pre-trained
version of T5 on ONNX included in the package.

> **_NOTE:_**  Please note that the first time you call get_encoder_decoder_tokenizer, the models are
being downloaded which might take a minute or two.



## Train your own MIST Estimators

We ship a simple CLI (mist-statinf) to reproduce the full workflow.

### 1. Data Generation
```bash
mist-statinf generate configs/generate/mini.yaml --version local
```

### 2. Train a MIST Model
```bash
mist-statinf train configs/train/mini.yaml
```

### 3. Hyper parameters search
```bash
mist-statinf generate configs/generate/mini.yaml --version local
```

### 3. Inference
```bash
RUN_DIR="logs/mist_mini/run_YYYYmmdd-HHMMSS"
mist-statinf infer configs/inference/mini_inf.yaml "$RUN_DIR" --out-path mi_results.json

# CSV predictions → $RUN_DIR/predictions_mi_mini_local.csv
# JSON summary   → mi_results.json
```
mode: point — one pass, fast. \
mode: bootstrap — bootstrap mean/var + percentile CI. \
mode: qcqr_calib — calibration for QCQR models (empirical coverage vs target quantile).

### 3. Run baselines
```bash
mist-statinf baselines configs/test/mini_test.yaml

# Results per estimator → logs/baselines_mini/run_YYYYmmdd-HHMMSS/*.csv
```
## References ##
If you use ```mist-statinf```, please consider citing:
```bibtex
@article{johnson2013hdphsmm,
    title={Bayesian Nonparametric Hidden Semi-Markov Models},
    author={Johnson, Matthew J. and Willsky, Alan S.},
    journal={Journal of Machine Learning Research},
    pages={673--701},
    volume={14},
    month={February},
    year={2013},
}
```

## Authors ##

[German Gritsai](https://github.com/), [Megan Richards](https://github.com/), [Maxime Meloux](https://github.com/), [Kyunghyun Cho](https://github.com/), [Maxime Peyrard](https://github.com/).