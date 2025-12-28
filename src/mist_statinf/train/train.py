# src/mist/train/train.py
from __future__ import annotations
import os, glob, shutil, datetime
from typing import Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor, RichProgressBar, StochasticWeightAveraging
)
from pytorch_lightning.loggers import CSVLogger
from omegaconf import OmegaConf, DictConfig

from mist_statinf.data.meta_dataloader import MetaStatDataModule   
from mist_statinf.train.lit_module import MISTModelLit             
from mist_statinf.utils.logging import setup_logging              

def _record_meta_distribution(config: DictConfig, output_dir: str) -> str | None:
    train_data_path = config.datamodule.train_folder
    yaml_files = glob.glob(os.path.join(train_data_path, "*.yaml"))
    if not yaml_files:
        return None
    if len(yaml_files) != 1:
        raise ValueError(f"Expected exactly one .yaml in {train_data_path}, found {len(yaml_files)}")
    dest_path = os.path.join(output_dir, "in_meta_distribution.yaml")
    shutil.copy(yaml_files[0], dest_path)
    return dest_path

def run(config: DictConfig) -> Dict[str, Any]:
    # --- logs
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = config.get("name", "experiment")
    output_dir = os.path.join("logs", exp_name, f"run_{timestamp}")
    logger = setup_logging(output_dir, config.get("log_file", "experiment.log"))
    logger.info(f"Output directory: {output_dir}")

    # --- save config
    out_cfg_path = os.path.join(output_dir, f"{exp_name}.yaml")
    OmegaConf.save(config, out_cfg_path)
    logger.info("Loaded configuration:\n" + OmegaConf.to_yaml(config))

    if config.get("seed") is not None:
        pl.seed_everything(config.get("seed"))

    try:
        meta_record = _record_meta_distribution(config, output_dir)
        if meta_record:
            logger.info(f"Recorded meta-distribution to: {meta_record}")
    except Exception as e:
        logger.warning(f"Meta-distribution record skipped: {e}")

    # --- DataModule
    logger.info("Instantiating DataModule")
    datamodule = MetaStatDataModule(**config["datamodule"])

    # --- Model
    logger.info("Instantiating Lightning model")
    model = MISTModelLit(config, output_filepath=os.path.join(output_dir, "test_predictions.jsonl"))

    # --- Callbacks
    logger.info("Instantiating callbacks")
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="best_model_{epoch}-{val_loss:.2f}",
        monitor="val_loss", mode="min", save_last=False, verbose=False
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    bar_cb = RichProgressBar(leave=True)

    callbacks = [ckpt_cb, lr_cb, bar_cb]
    swa_lr = config.get("optimizer", {}).get("swa_lr", None)
    if swa_lr:
        logger.info("Enabling Stochastic Weight Averaging")
        callbacks.append(StochasticWeightAveraging(swa_lrs=swa_lr))

    # --- Logger
    pl_logger = CSVLogger(save_dir=output_dir, name="pl_CSVLogger")

    # --- Trainer
    logger.info("Instantiating Trainer")
    trainer = pl.Trainer(
        **config["trainer"],
        callbacks=callbacks,
        logger=[pl_logger],
        use_distributed_sampler=False,
    )

    # --- Train
    logger.info("Training started")
    t0 = datetime.datetime.now()
    trainer.fit(model=model, datamodule=datamodule)
    dt = str(datetime.datetime.now() - t0)

    # --- Best checkpoint
    best_ckpt = ckpt_cb.best_model_path
    logger.info(f"Best checkpoint: {best_ckpt}")

    # --- Params count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {num_params:,}")

    # --- Final test on best
    logger.info("Testing best checkpoint")
    best_model = MISTModelLit.load_from_checkpoint(best_ckpt, args=config, weights_only=False)
    test_results = trainer.test(best_model, datamodule=datamodule)

    # --- Summary
    summary = {
        "Best Model Checkpoint": best_ckpt,
        "Total Training Time": dt,
        "Number of Parameters": int(num_params),
        "Test Results": test_results[0] if test_results else {},
    }
    summary_path = os.path.join(output_dir, "training_summary.yaml")
    OmegaConf.save(OmegaConf.create(summary), summary_path)
    logger.info("Training Summary:\n" + OmegaConf.to_yaml(summary))
    return summary

def train_main(config_path: str, ckpt_out: str = "checkpoints") -> None:
    cfg = OmegaConf.load(config_path)
    run(cfg)
