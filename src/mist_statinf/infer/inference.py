from __future__ import annotations

import os
import glob
import json
import time
import math
import copy
import tqdm
from typing import Dict, Tuple, Iterable, List, Any

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig

from mist_statinf.train.lit_module import MISTModelLit
from mist_statinf.utils import resolve_device, pad_truncate, bootstrap_data, load_meta_dataset


# ------------------------ helper ------------------------


def _load_model_from_run_dir(run_dir: str) -> Tuple[MISTModelLit, DictConfig]:
    """
    Loads the best Lightning checkpoint and its hparams from a training run directory.
    Expects:
      - {run_dir}/checkpoints/best_model_*
      - {run_dir}/pl_CSVLogger/version_0/*.yaml
    """
    ckpts = glob.glob(os.path.join(run_dir, "*.ckpt"))
    if len(ckpts) != 1:
        raise ValueError(f"Expected exactly one best checkpoint, found {len(ckpts)}: {ckpts}")
    best_ckpt = ckpts[0]

    hp = glob.glob(os.path.join(run_dir, "pl_CSVLogger", "version_0", "*.yaml"))
    if len(hp) != 1:
        raise ValueError(f"Expected exactly one hparams yaml, found {len(hp)}: {hp}")
    hparams = OmegaConf.load(hp[0])

    model = MISTModelLit.load_from_checkpoint(best_ckpt, args=hparams.args)
    model.eval()
    return model, hparams



# ------------------------ bootstrap & QCQR routines ------------------------


@torch.no_grad()
def _get_bootstrapped_predictions(
    meta_dataset: Iterable[dict],
    estimator: torch.nn.Module,
    device: torch.device,
    n_resamples: int = 10,
    alpha: float = 0.05,
    tau_value: float | None = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Bootstrap-based analysis: for each meta-datapoint we estimate the sampling
    distribution of MI via resampling and compute Bias, MSE, Variance, CI and coverage.
    """
    predictions: List[dict] = []
    times: List[float] = []

    target_feature_dim = estimator.phi.embedding_mlp[0].in_features

    for md in tqdm.tqdm(meta_dataset, desc="Bootstrap", leave=False):
        arr = md["source"]  # (N, D)
        tgt = float(md["target"][0] if np.ndim(md["target"]) else md["target"])

        ests: List[float] = []

        t0 = time.time()
        for bs in bootstrap_data(arr, n_resamples):
            x = torch.tensor(bs, dtype=torch.float32, device=device).unsqueeze(0)
            x = pad_truncate(x, target_feature_dim)

            batch = {"source": x}
            if tau_value is not None:
                batch["tau"] = torch.full((1, 1), float(tau_value), device=device)

            out = estimator(batch).item()
            if not (math.isnan(out) or math.isinf(out)):
                ests.append(float(out))
        times.append(time.time() - t0)

        if len(ests) == 0:
            ests = [float("nan")]

        mean_est = float(np.nanmean(ests))
        var_est = float(np.nanvar(ests))
        lower = float(np.nanpercentile(ests, 100 * (alpha / 2)))
        upper = float(np.nanpercentile(ests, 100 * (1 - alpha / 2)))
        cov = float(lower <= tgt <= upper)
        mse = float(np.nanmean([(e - tgt) ** 2 for e in ests]))
        bias2 = float((mean_est - tgt) ** 2)

        rec = copy.deepcopy(md)
        rec.pop("source", None)
        rec.update({
            "output": ests,
            "Bias": bias2,
            "MSE": mse,
            "Variance": var_est,
            "ci_lower": lower,
            "ci_upper": upper,
            "CI cov.": cov,
            "target": tgt,
        })
        predictions.append(rec)

    avg_time = float(np.mean(times)) if times else float("nan")
    return pd.DataFrame(predictions), avg_time


@torch.no_grad()
def _get_qcqr_empirical_coverage(
    meta_dataset: Iterable[dict],
    estimator: torch.nn.Module,
    device: torch.device,
    quantiles_to_run: np.ndarray,
    n_resamples: int = 30,
) -> Tuple[pd.DataFrame, float]:
    """
    QCQR calibration analysis.

    For each meta-datapoint, we:
      1) Build a bootstrap distribution of the (τ = 0.5) QCQR estimate.
      2) For each target quantile τ:
           - compute the predicted τ-quantile q_τ(x),
           - compute empirical coverage = P[bootstrap_value <= q_τ(x)].

    Used to assess calibration quality of QCQR models.
    """
    results: List[dict] = []
    times: List[float] = []

    target_feature_dim = estimator.phi.embedding_mlp[0].in_features

    for md in tqdm.tqdm(meta_dataset, desc="QCQR calibration", leave=False):
        arr = md["source"]

        t0 = time.time()

        # 1) bootstrap distribution at tau = 0.5
        bs_vals: List[float] = []
        for bs in bootstrap_data(arr, n_resamples):
            x = torch.tensor(bs, dtype=torch.float32, device=device).unsqueeze(0)
            x = pad_truncate(x, target_feature_dim)
            batch = {"source": x, "tau": torch.full((1, 1), 0.5, device=device)}
            val = estimator(batch).item()
            if not (math.isnan(val) or math.isinf(val)):
                bs_vals.append(float(val))

        bs_np = np.array(bs_vals, dtype=float)

        # 2) for each target quantile, compute predicted q_tau and empirical coverage
        for tau in quantiles_to_run:
            x0 = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
            x0 = pad_truncate(x0, target_feature_dim)
            batch0 = {"source": x0, "tau": torch.full((1, 1), float(tau), device=device)}

            q_pred = estimator(batch0).item()
            coverage = float(np.mean(bs_np <= q_pred)) if bs_np.size > 0 else float("nan")

            rec = copy.deepcopy(md)
            rec.pop("source", None)
            rec.update({
                "target_quantile": float(tau),
                "predicted_quantile_value": float(q_pred),
                "empirical_coverage": coverage,
            })
            results.append(rec)

        times.append(time.time() - t0)

    avg_time = float(np.mean(times)) if times else float("nan")
    return pd.DataFrame(results), avg_time


# ------------------------------ main CLI entry ------------------------------


def infer_main(config_path: str, ckpt_dir: str, out_path: str = "mi_results.json") -> None:
    """
    Main entry point for CLI command `mist-statinf infer`.
    """
    cfg: DictConfig = OmegaConf.load(config_path)
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))

    if cfg.get("seed"):
        pl.seed_everything(cfg.get("seed"))

    device = resolve_device(cfg.get("device", "auto"))
    print(f"Device: {device}")

    # Load model and hparams from run directory
    model, hparams = _load_model_from_run_dir(ckpt_dir)
    model.to(device).eval()

    # Load meta-dataset
    meta = load_meta_dataset(cfg.dataset_path)

    # Optional dimension filter
    filter_dim = cfg.get("datamodule", {}).get("filter_dim", None)
    if filter_dim is not None:
        before = len(meta)
        meta = [dp for dp in meta if dp.get("dimension") == filter_dim]
        print(f"Applied filter_dim={filter_dim}: {len(meta)} / {before} items kept.")
        if not meta:
            print("No data after filtering; exiting.")
            return

    # Is it a QCQR model?
    is_qcqr = getattr(model.msm, "quantile_conditioned", False)

    # Unpack infer section
    infer_cfg = cfg.get("infer", {}) or {}
    mode = str(infer_cfg.get("mode", "bootstrap")).lower()
    n_resamples = int(infer_cfg.get("n_resamples", 30))
    alpha = float(infer_cfg.get("alpha", 0.05))
    tau = infer_cfg.get("tau", None)
    quantiles = infer_cfg.get("quantiles", None)
    if quantiles is None and mode == "qcqr_calib":
        quantiles = [round(x, 2) for x in np.linspace(0.05, 0.95, 19).tolist()]

    dataset_name = os.path.basename(os.path.normpath(cfg.dataset_path))
    csv_out_path = os.path.join(ckpt_dir, f"predictions_{dataset_name}.csv")

    results_dict: Dict[str, Any] = {}

    if mode == "bootstrap":
        print("Running BOOTSTRAP inference.")
        # For QCQR, we can choose which quantile to bootstrap (e.g., median tau=0.5)
        tau_used = (float(tau) if (is_qcqr and tau is not None) else (0.5 if is_qcqr else None))
        df, tavg = _get_bootstrapped_predictions(
            meta_dataset=meta,
            estimator=model.msm,
            device=device,
            n_resamples=n_resamples,
            alpha=alpha,
            tau_value=tau_used,
        )
        summary = {
            "mean_mse": float(np.nanmean(df["MSE"])),
            "mean_bias_squared": float(np.nanmean(df["Bias"])),
            "mean_variance": float(np.nanmean(df["Variance"])),
            "ci_coverage": float(np.nanmean(df["CI cov."])),
        }
        df.to_csv(csv_out_path, index=False)
        results_dict = {
            "mode": "bootstrap",
            "is_qcqr": bool(is_qcqr),
            "tau_used": tau_used,
            "n_resamples": n_resamples,
            "alpha": alpha,
            "summary_stats": summary,
            "execution_time_avg_s": float(tavg),
        }
        print(f"Saved bootstrap predictions to {csv_out_path}")

    elif mode == "qcqr_calib":
        if not is_qcqr:
            raise ValueError("qcqr_calib mode requires a QCQR model (quantile_conditioned=True).")
        print("Running QCQR CALIBRATION.")
        qs = np.array(quantiles, dtype=float)
        df, tavg = _get_qcqr_empirical_coverage(
            meta_dataset=meta,
            estimator=model.msm,
            device=device,
            quantiles_to_run=qs,
            n_resamples=n_resamples,
        )
        calib_summary = df.groupby("target_quantile")["empirical_coverage"].mean().reset_index()
        mae = float(np.mean(np.abs(
            calib_summary["target_quantile"] - calib_summary["empirical_coverage"]
        )))
        df.to_csv(csv_out_path, index=False)
        results_dict = {
            "mode": "qcqr_calib",
            "is_qcqr": True,
            "n_resamples": n_resamples,
            "quantiles": qs.tolist(),
            "calibration_mae": mae,
            "execution_time_avg_s": float(tavg),
        }
        print(f"Saved QCQR calibration table to {csv_out_path}")

    else:
        raise ValueError(
            f"Unknown infer.mode='{mode}'. "
            "Use one of: bootstrap | qcqr_calib. "
            "For point predictions, use MISTQuickEstimator."
        )

    # JSON summary
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Wrote JSON summary to {out_path}")