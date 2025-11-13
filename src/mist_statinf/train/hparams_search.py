from __future__ import annotations
import os, glob, json, logging, math
from functools import partial
from typing import Dict, Any, Optional

import optuna
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from mist_statinf.data.meta_dataloader import MetaStatDataModule
from mist_statinf.train.lit_module import MISTModelLit
from mist_statinf.utils.logging import setup_logging

# ----------------------------- Helpers -----------------------------------------

def _find_one(pattern: str, err: str) -> str:
    paths = glob.glob(pattern)
    if len(paths) != 1:
        raise ValueError(f"{err}: found {len(paths)} -> {paths}")
    return paths[0]

def load_hparams_from_run_dir(run_dir: str) -> DictConfig:
    """
    Loads the single hparams YAML from a Lightning run dir:
      {run_dir}/pl_CSVLogger/version_0/*.yaml
    """
    hp = _find_one(
        os.path.join(run_dir, "pl_CSVLogger", "version_0", "*.yaml"),
        "Expected exactly one hparams yaml"
    )
    return OmegaConf.load(hp)

def load_best_ckpt_path(run_dir: str) -> str:
    """Loads the best checkpoint path from a Lightning run dir."""
    return _find_one(
        os.path.join(run_dir, "checkpoints", "best_model_*"),
        "Expected exactly one best checkpoint"
    )

def merge_args(base: DictConfig, patch: Dict[str, Any]) -> DictConfig:
    """Deep-merge dict patch into base OmegaConf DictConfig."""
    return OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(patch))

def _as_float(val) -> float:
    try:
        return float(val.item())  # tensor
    except Exception:
        return float(val)

def _device_hint() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ------------------------- Objective (Optuna) -----------------------------------

def _objective(
    trial: optuna.trial.Trial,
    base_args: DictConfig,
    out_dir: str,
    model_type: Optional[str] = None,
    monitor_metric: str = "val_loss",
) -> float:
    """
    Single Optuna trial:
      - samples hyperparams
      - builds args (deep merge: base + sampled)
      - trains for few epochs with EarlyStopping
      - returns monitored validation metric
    """
    logger = logging.getLogger("optuna.trial")
    seed = int(base_args.get("seed", 1234))
    seed_everything(seed, workers=True)

    # Decide loss type: explicit override wins; else keep from base
    loss_type = (model_type or base_args.get("loss_type", "MSE")).upper()
    is_qcqr = (loss_type == "QCQR")

    # --- Search space (compact, extend as needed) ---
    lr            = trial.suggest_float("optimizer.lr", 1e-5, 2e-4, log=True)
    weight_decay  = trial.suggest_float("optimizer.weight_decay", 1e-6, 1e-4, log=True)
    phi_dim_fwd   = trial.suggest_categorical("architecture.phi_dim_forward", [256, 384, 512, 768, 1024])
    n_phi_layers  = trial.suggest_int("architecture.n_phi_layers", 2, 4)
    n_dec_layers  = trial.suggest_int("architecture.n_dec_layers", 1, 2)
    n_rho_layers  = trial.suggest_int("architecture.n_rho_layers", 1, 3)
    n_phi_heads   = trial.suggest_categorical("architecture.n_phi_heads", [4, 8, 16])
    n_inds        = trial.suggest_categorical("architecture.n_inds", [16, 32, 64])
    n_seeds       = trial.suggest_categorical("architecture.n_seeds", [3, 5, 10, 15])

    # Patch with sampled values (and keep everything else from base)
    patch: Dict[str, Any] = {
        "loss_type": loss_type,
        "architecture": {
            "phi_dim_forward": phi_dim_fwd,
            "n_phi_layers": n_phi_layers,
            "n_dec_layers": n_dec_layers,
            "n_rho_layers": n_rho_layers,
            "n_phi_heads": n_phi_heads,
            "n_inds": n_inds,
            "n_seeds": n_seeds,
            "quantile_conditioned": is_qcqr,
        },
        "optimizer": {
            "lr": lr,
            "weight_decay": weight_decay,
            "scheduler": {
                "name": base_args.optimizer.scheduler.get("name", "on_plateau"),
                "mode": base_args.optimizer.scheduler.get("mode", "min"),
                "metric": base_args.optimizer.scheduler.get("metric", "val_loss"),
                "patience": base_args.optimizer.scheduler.get("patience", 3),
                "min_lr": base_args.optimizer.scheduler.get("min_lr", 5e-6),
            },
        },
    }
    args = merge_args(base_args, patch)

    # Datamodule (use paths from base/hparams)
    dm = MetaStatDataModule(**args["datamodule"])

    # Model
    model = MISTModelLit(args, output_filepath=os.path.join(out_dir, "trial_preds.jsonl"))

    # Logging per-trial
    csv_logger = CSVLogger(save_dir=out_dir, name=f"trial_{trial.number}")
    early_stop = EarlyStopping(monitor=monitor_metric, patience=3, mode="min")

    # Trainer (lean; disable progress bar for speed)
    trainer = Trainer(
        max_epochs=int(args.get("trainer", {}).get("max_epochs", 30)),
        gradient_clip_val=float(args.get("trainer", {}).get("gradient_clip_val", 0.5)),
        precision=int(args.get("trainer", {}).get("precision", 16)),
        enable_checkpointing=False,
        logger=csv_logger,
        callbacks=[early_stop],
        enable_model_summary=False,
        log_every_n_steps=50,
        # keep accelerator/devices in args if present; otherwise hint
        accelerator=args.get("trainer", {}).get("accelerator", _device_hint()),
        devices=args.get("trainer", {}).get("devices", 1),
        detect_anomaly=False,
    )

    # Hook pruning to val metric (after each epoch)
    prune_cb_metric = monitor_metric

    try:
        trainer.fit(model, dm)
        metric = trainer.callback_metrics.get(prune_cb_metric)
        if metric is None or not math.isfinite(_as_float(metric)):
            # If metric missing, treat as bad trial
            raise optuna.TrialPruned(f"Missing/invalid metric '{prune_cb_metric}'.")

        score = _as_float(metric)
        trial.report(score, step=trainer.current_epoch or 0)
        # Optionally prune if not improving
        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {trainer.current_epoch}.")
        logger.info("Trial %d done: %s=%.6f", trial.number, prune_cb_metric, score)
        return score

    except optuna.TrialPruned as e:
        logger.warning("Trial %d pruned: %s", trial.number, str(e))
        raise
    except Exception as e:
        logger.exception("Trial %d failed: %s", trial.number, str(e))
        # Return a large score so it's clearly worse than successful trials
        return float("inf")


# ------------------------------- Public API -------------------------------------

def build_base_args_from_run(run_dir: str) -> DictConfig:
    """
    Create a BASE args DictConfig from a previous training run directory
    (uses its hparams.yaml).

    Expected:
      - {run_dir}/pl_CSVLogger/version_0/*.yaml
    """
    hparams = load_hparams_from_run_dir(run_dir)
    if "args" in hparams:
        return hparams["args"]
    return hparams 


def hparam_search(
    base_args: DictConfig,
    out_dir: str = os.path.join("logs", "params_tuning"),
    model_type: Optional[str] = None,           # "MSE" | "QCQR" | None -> use base
    n_trials: int = 50,
    timeout_sec: Optional[int] = None,          # wall-clock timeout for study
    study_name: Optional[str] = None,
    monitor_metric: str = "val_loss",
    seed: int = 42,
) -> optuna.study.Study:
    """
    Run Optuna hyper-parameter search using a given base args config.

    base_args:
        OmegaConf DictConfig (e.g., from a prior run's hparams 'args').
    out_dir:
        Directory to store trial CSV logs and a summary json/yaml.
    model_type:
        Override loss type ("MSE" / "QCQR") or None to keep base.

    Returns:
        optuna.study.Study (best params accessible via study.best_trial)
    """
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(out_dir, "params_search_info.log")
    logger = logging.getLogger("tune")

    # Deterministic sampler/pruner
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=2, interval_steps=1
    )

    # Use JournalStorage to resume across runs
    storage = optuna.storages.journal.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            os.path.join(out_dir, f"optuna_{(model_type or base_args.get('loss_type','MSE'))}.log")
        )
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        storage=storage,
        study_name=study_name or f"MIST_{(model_type or base_args.get('loss_type', 'MSE'))}",
    )

    obj = partial(
        _objective,
        base_args=base_args,
        out_dir=out_dir,
        model_type=model_type,
        monitor_metric=monitor_metric,
    )

    logger.info("Starting study '%s' with %d trials (timeout=%s)",
                study.study_name, n_trials, str(timeout_sec))
    study.optimize(obj, n_trials=n_trials, n_jobs=1, timeout=timeout_sec)

    # Persist results
    results_json = os.path.join(out_dir, "study_summary.json")
    results_yaml = os.path.join(out_dir, "best_params.yaml")
    with open(results_json, "w") as f:
        json.dump(
            {
                "study_name": study.study_name,
                "best_value": study.best_value,
                "best_params": study.best_trial.params,
                "best_trial_number": study.best_trial.number,
                "n_trials": len(study.trials),
            }, f, indent=2
        )
    OmegaConf.save(OmegaConf.create(study.best_trial.params), results_yaml)

    logger.info("Best value: %.6f", study.best_value)
    logger.info("Best params saved to: %s", results_yaml)
    logger.info("Study summary saved to: %s", results_json)
    return study


# ------------------------------- Convenience CLI -------------------------------

def hparam_main(
    run_dir: Optional[str] = None,     
    model_type: str = "QCQR",
    n_trials: int = 50,
    timeout_sec: Optional[int] = None,
):
    """
    Convenience wrapper for quick tuning from an existing run directory:
      - Reads {run_dir}/pl_CSVLogger/version_0/*.yaml to form base args
      - Runs optuna search and stores outputs in logs/params_tuning/
    """
    out_dir = os.path.join("logs", "params_tuning")
    if run_dir:
        base_args = build_base_args_from_run(run_dir)
    else:
        raise ValueError("Please provide run_dir to seed base hyper-parameters.")

    hparam_search(
        base_args=base_args,
        out_dir=out_dir,
        model_type=model_type,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
    )