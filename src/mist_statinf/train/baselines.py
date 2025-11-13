from __future__ import annotations
import argparse
import copy
import datetime as dt
import math
import os
import pickle
import time
from typing import Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import tqdm
from omegaconf import OmegaConf, DictConfig

import bmi  

from mist_statinf.utils.logging import setup_logging
from mist_statinf.utils.bootstrap import bootstrap_data 

# Possible baselines
ESTIMATORS = {
    "CCA": bmi.estimators.CCAMutualInformationEstimator(),
    "KSG": bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    "MINE": bmi.estimators.MINEEstimator(batch_size=4),
    "InfoNCE": bmi.estimators.InfoNCEEstimator(batch_size=4),
    "NWJE": bmi.estimators.NWJEstimator(batch_size=5),
}


def _get_bootstrapped_predictions(meta_dataset, estimator, n_resamples: int = 7, alpha: float = 0.05
                                  ) -> Tuple[pd.DataFrame, float]:
    """
    Compute bootstrap-based MI estimates and confidence intervals for a meta-dataset.

    Args:
        meta_dataset: Iterable of dicts with keys:
            - "source": np.ndarray of shape (N, 2) containing paired samples (X, Y).
            - "target": float or array-like with ground-truth MI (first element is used if array-like).
        estimator: Baseline MI estimator exposing .estimate(X: np.ndarray, Y: np.ndarray) -> float.
        n_resamples (int): Number of bootstrap resamples per meta-datapoint. Defaults to 7.
        alpha (float): Two-sided significance level for percentile CI (e.g., 0.05 → 95% CI). Defaults to 0.05.

    Returns:
        Tuple[pandas.DataFrame, float]:
            - DataFrame with one row per meta-datapoint containing:
                * output: list[float] — bootstrap MI estimates (NaN filtered).
                * Bias: float — squared bias ( (mean_est - target)^2 ).
                * MSE: float — mean squared error over bootstrap estimates.
                * Variance: float — variance of bootstrap estimates.
                * ci_lower / ci_upper: float — percentile CI bounds at alpha/2 and 1-alpha/2.
                * CI cov.: float — indicator (0/1) whether target lies within CI.
                * target: float — ground-truth MI.
                * plus any original fields from meta_datapoint except "source".
            - avg_runtime: float — average wall-clock time per single bootstrap estimate (seconds).

    Notes:
        - Each bootstrap resample `arr` is expected to be shape (N, 2); columns are split into X and Y.
        - NaN/Inf estimates are dropped before statistics; if all are invalid, a single NaN is kept.
        - CI is the simple percentile CI over bootstrap estimates.
    """
    predictions = []
    run_times = []

    for meta_datapoint in tqdm.tqdm(meta_dataset, desc="Baselines", leave=False):
        dataset = meta_datapoint["source"]
        target_mi = float(meta_datapoint["target"][0] if np.ndim(meta_datapoint["target"]) else meta_datapoint["target"])

        estimates = []
        for arr in bootstrap_data(dataset, n_resamples):
            # We expect arr to have the form (N, 2) -> we split it into X, Y
            X_res = arr[:, 0].reshape(-1, 1)
            Y_res = arr[:, 1].reshape(-1, 1)

            t0 = time.time()
            estimated_mi = estimator.estimate(X_res, Y_res)
            run_times.append(time.time() - t0)

            if not (math.isnan(estimated_mi) or math.isinf(estimated_mi)):
                estimates.append(estimated_mi)

        if len(estimates) == 0:
            estimates = [float("nan")]


        mean_est = float(np.nanmean(estimates))
        var_est = float(np.nanvar(estimates))
        lower = float(np.nanpercentile(estimates, 100 * (alpha / 2)))
        upper = float(np.nanpercentile(estimates, 100 * (1 - alpha / 2)))
        coverage = (lower <= target_mi) and (target_mi <= upper)

        rec = copy.deepcopy(meta_datapoint)
        rec.pop("source", None) 
        rec["output"] = estimates
        rec["Bias"] = (mean_est - target_mi) ** 2
        rec["MSE"] = float(np.nanmean([(est - target_mi) ** 2 for est in estimates]))
        rec["Variance"] = var_est
        rec["ci_lower"] = lower
        rec["ci_upper"] = upper
        rec["CI cov."] = float(coverage)
        rec["target"] = target_mi

        predictions.append(rec)

    avg_runtime = float(np.mean(run_times)) if run_times else float("nan")
    return pd.DataFrame(predictions), avg_runtime


def run(config: DictConfig) -> None:
    # Logs
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = config.get("name", "baselines")
    output_dir = os.path.join("logs", exp_name, f"baselines_{timestamp}")
    logger = setup_logging(output_dir, config.get("log_file", "baselines.log"))
    logger.info(f"Output directory: {output_dir}")

    # Save config
    OmegaConf.save(config, os.path.join(output_dir, f"{exp_name}.yaml"))
    logger.info("Loaded configuration:\n" + OmegaConf.to_yaml(config))

    # Seed
    if config.get("seed") is not None:
        pl.seed_everything(config.get("seed"))

    # Test loading
    test_folder = config.data.test_folder
    meta_path = os.path.join(test_folder, "meta_dataset.pkl")
    logger.info(f"Loading meta-dataset: {meta_path}")
    with open(meta_path, "rb") as f:
        test_meta_dataset = pickle.load(f)

    # Params for CI
    alpha = float(config.get("ci", {}).get("alpha", 0.05))
    n_resamples = int(config.get("ci", {}).get("n_resamples", 7))

    to_run = set(config.get("estimators_to_run", list(ESTIMATORS.keys())))

    for name, estimator in ESTIMATORS.items():
        if name not in to_run:
            logger.info(f"Estimator {name}: skipped")
            continue

        logger.info(f"Estimator {name}: running with n_resamples={n_resamples}, alpha={alpha}")
        df, exec_time = _get_bootstrapped_predictions(test_meta_dataset, estimator, n_resamples=n_resamples, alpha=alpha)
        df["run_time"] = exec_time

        out_csv = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(out_csv, index=False)
        logger.info(
            f"{name}: avg_time={exec_time:.4f}s "
            f"| TEST MSE={np.nanmean(df['MSE']):.6f} "
            f"| Bias={np.nanmean(df['Bias']):.6f} "
            f"| Var={np.nanmean(df['Variance']):.6f} "
            f"| CI cov.={np.nanmean(df['CI cov.']):.3f}"
        )

    logger.info(f"Baseline results saved to: {output_dir}")


def baselines_main(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    run(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline MI estimators")
    parser.add_argument("--config", required=True, help="Path to config YAML (e.g., configs/test/mist_test.yaml)")
    args = parser.parse_args()
    baselines_main(args.config)
