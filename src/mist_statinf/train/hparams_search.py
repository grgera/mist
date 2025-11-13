from __future__ import annotations
import os, logging
from functools import partial
from typing import Dict, Any

import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from mist_statinf.data.meta_dataloader import MetaStatDataModule
from mist_statinf.train.lit_module import MISTModelLit
from mist_statinf.utils.logging import setup_logging

def _objective(trial: optuna.trial.Trial, model_type: str) -> float:
    logger = logging.getLogger("optuna_trial")
    logger.info("Starting new trial... Model type: %s", model_type)

    # --- search space
    lr = trial.suggest_float('lr', 1e-5, 2e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    phi_dim_forward = trial.suggest_categorical('phi_dim_forward', [256, 384, 512, 768, 1024])
    n_phi_layers = trial.suggest_int('n_phi_layers', 2, 4)
    n_dec_layers = trial.suggest_int('n_dec_layers', 1, 2)
    n_rho_layers = trial.suggest_int('n_rho_layers', 1, 3)
    n_phi_heads = trial.suggest_categorical('n_phi_heads', [4, 8, 16])
    n_inds = trial.suggest_categorical('n_inds', [16, 32, 64])
    n_seeds = trial.suggest_categorical('n_seeds', [3, 5, 10, 15])

    args: Dict[str, Any] = {
        "loss_type": model_type.upper(),  # "MSE" | "QCQR"
        "architecture": {
            "max_input_dim": 64,
            "n_phi_layers": n_phi_layers,
            "n_dec_layers": n_dec_layers,
            "phi_hidden_dim": 256,
            "n_phi_heads": n_phi_heads,
            "phi_dim_forward": phi_dim_forward,
            "phi_activation_fun": "gelu",
            "n_rho_layers": n_rho_layers,
            "rho_hidden_dim": 256,
            "n_inds": n_inds,
            "output_dim": 1,
            "phi_model": "set_transformer",
            "quantile_conditioned": (model_type.lower() == "qcqr"),
            "sab_stack_layers": 2,
            "n_seeds": n_seeds,
        },
        "optimizer": {
            "eps": 8e-9,
            "lr": lr,
            "swa_lr": None,
            "weight_decay": weight_decay,
            "scheduler": {
                "name": "on_plateau",
                "mode": "min",
                "metric": "train_loss",
                "patience": 3,
                "min_lr": 5e-6
            }
        },
        "datamodule": {
            "train_folder": "data/train_data",
            "val_folder": "data/val_grid",
            "test_folder": "data/test_imd_data_grid",
            "batch_size": 512
        },
        "trainer": {
            "gradient_clip_val": 0.5,
            "max_epochs": 30,
            "precision": 16,
            "enable_checkpointing": False
        }
    }

    model = MISTModelLit(args, output_filepath=os.path.join("logs","params_tuning","test_predictions.jsonl"))
    datamodule = MetaStatDataModule(**args["datamodule"])

    csv_logger = CSVLogger(save_dir=os.path.join("logs","params_tuning"), name=f"trial_{trial.number}")
    early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    trainer = Trainer(
        logger=csv_logger,
        callbacks=[early_stop],
        gradient_clip_val=args["trainer"]["gradient_clip_val"],
        max_epochs=args["trainer"]["max_epochs"],
        precision=args["trainer"]["precision"],
        enable_checkpointing=args["trainer"]["enable_checkpointing"]
    )

    trainer.fit(model, datamodule)
    val = trainer.callback_metrics.get("val_loss")
    val_loss = float(val.item()) if val is not None else float("inf")
    logger.info(f"Trial {trial.number} finished. Val_loss={val_loss:.6f}")
    return val_loss

def hparam_main(model_type: str = "QCQR", n_trials: int = 50):
    out_dir = os.path.join("logs", "params_tuning")
    setup_logging(out_dir, "params_search_info.log")
    logger = logging.getLogger("optuna")

    storage = optuna.storages.journal.JournalStorage(
        optuna.storages.journal.JournalFileBackend(f"./optuna_{model_type}.log")
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1),
        load_if_exists=True,
        storage=storage,
        study_name=f"MIST_{model_type}"
    )

    obj = partial(_objective, model_type=model_type)
    study.optimize(obj, n_trials=n_trials, n_jobs=1)

    logger.info(f"Best trial params: {study.best_trial.params}")
    logger.info(f"Best trial score: {study.best_value}")
