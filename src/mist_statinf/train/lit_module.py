from typing import Any, Dict
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from mist_statinf.model.estimator import MISTModel
from mist_statinf.model.losses import pinball_loss  


class MISTModelLit(pl.LightningModule):
    """
    PyTorch Lightning wrapper around the MIST estimator (transformer-based MI model).

    Args:
        args (DictConfig):
            Hydra/OmegaConf configuration with sections:
            - loss_type (str): "MSE" or "QCQR".
            - architecture (dict): kwargs for `MISTModel(**architecture)`.
            - optimizer (dict):
                * lr (float), eps (float), weight_decay (float)
                * scheduler (dict, optional):
                    - name: "cosine" | "on_plateau" | None
                    - mode (str, optional): e.g., "min" (for on_plateau)
                    - patience (int, optional)
                    - min_lr (float, optional)
                    - metric (str, optional): monitored metric (default: "val_loss")
            - trainer (dict): used for scheduler horizon (e.g., max_epochs).
        output_filepath (str | None, optional):
            Optional path for dumping predictions/metrics outside the logger.

    Returns:
        - `forward(batch)`: torch.Tensor — model outputs (shape depends on loss_type; typically (B, 1)).
        - `training_step(...)`: torch.Tensor — scalar loss.
        - `validation_step(...)`: torch.Tensor — scalar val loss (averaged for QCQR).
        - `test_step(...)`: torch.Tensor — scalar test loss (averaged for QCQR).
    """
    def __init__(self, args: DictConfig, output_filepath: str | None = None):
        super().__init__()
        self.args = args
        self.output_filepath = output_filepath

        # --- Models -------------------------------------------------------------
        self.msm = MISTModel(**args["architecture"])
        self.loss_type = args["loss_type"]

        # --- Losses --------------------------------------------------------------
        if self.loss_type == "MSE":
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == "QCQR":
            self.loss_fn = pinball_loss
        else:
            raise ValueError(f"Loss type not supported: {self.loss_type}")

        self.save_hyperparameters(ignore=["output_filepath"])

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.msm(batch)

    def compute_loss(self, batch: Dict[str, Any], compute_mse: bool = False) -> Dict[str, Any]:
        y = batch["target"]

        if self.loss_type == "MSE":
            logits = self.forward(batch).squeeze(-1)
            loss = self.loss_fn(logits, y.float())
            mse_loss = None

        elif self.loss_type == "QCQR":
            batch_size = y.size(0)
            tau = torch.rand(batch_size, 1, device=self.device)
            batch_with_tau = dict(batch)
            batch_with_tau["tau"] = tau
            preds = self.forward(batch_with_tau).view(-1, 1)
            target = y.float().view(-1, 1)
            loss = self.loss_fn(preds, target, tau)
            logits, mse_loss = preds, None

        else:
            raise ValueError(f"Loss type not supported: {self.loss_type}")

        return {"loss": loss, "mse_loss": mse_loss, "preds": logits, "targets": y}

    def training_step(self, batch, batch_idx):
        outputs = self.compute_loss(batch)
        self.log("train_loss", outputs["loss"], on_step=True, prog_bar=True)
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        if self.loss_type == "QCQR":
            quantiles = torch.tensor([0.1, 0.5, 0.9], device=self.device)
            total = 0.0
            for q in quantiles:
                tau = torch.full((batch["target"].size(0), 1), q.item(), device=self.device)
                batch_q = dict(batch)
                batch_q["tau"] = tau
                preds = self.forward(batch_q).view(-1, 1)
                loss_q = self.loss_fn(preds, batch["target"].float().view(-1, 1), tau)
                self.log(f"val_loss_q{int(q.item()*100)}", loss_q, on_epoch=True)
                total += loss_q
            avg = total / len(quantiles)
            self.log("val_loss", avg, on_epoch=True, prog_bar=True)
            return avg
        else:
            outputs = self.compute_loss(batch, compute_mse=True)
            self.log("val_loss", outputs["loss"], on_epoch=True, prog_bar=True)
            return outputs["loss"]

    def test_step(self, batch, batch_idx):
        if self.loss_type == "QCQR":
            quantiles = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=self.device)
            total = 0.0
            for q in quantiles:
                tau = torch.full((batch["target"].size(0), 1), q.item(), device=self.device)
                batch_q = dict(batch)
                batch_q["tau"] = tau
                preds = self.forward(batch_q).view(-1, 1)
                loss_q = self.loss_fn(preds, batch["target"].float().view(-1, 1), tau)
                self.log(f"test_loss_q{int(q.item()*100)}", loss_q, on_epoch=True)
                total += loss_q
            avg = total / len(quantiles)
            self.log("test_loss", avg, on_epoch=True, prog_bar=True)
            return avg
        else:
            outputs = self.compute_loss(batch, compute_mse=True)
            loss = outputs["loss"] if self.loss_type == "MSE" else outputs["mse_loss"]
            self.log("test_loss", loss, on_epoch=True, prog_bar=True)
            return loss

    # ------------------------- Optimizers / Schedulers --------------------------

    def configure_optimizers(self):
        decay_params, no_decay_params = [], []
        for name, p in self.msm.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".bias") or p.ndim == 1:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        opt_cfg = self.args["optimizer"]
        optimizer_grouped_parameters = [
            {"params": decay_params,    "weight_decay": opt_cfg["weight_decay"], "eps": opt_cfg["eps"], "lr": opt_cfg["lr"]},
            {"params": no_decay_params, "weight_decay": 0.0,                      "eps": opt_cfg["eps"], "lr": opt_cfg["lr"]},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        sched_cfg = opt_cfg.get("scheduler", {"name": None})
        name = (sched_cfg.get("name") or "").lower()
        if name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.args["trainer"]["max_epochs"])
            return [optimizer], [scheduler]
        if name == "on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=sched_cfg.get("mode", "min"),
                patience=sched_cfg.get("patience", 5),
                min_lr=sched_cfg.get("min_lr", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": sched_cfg.get("metric", "val_loss"),
                    "interval": "epoch",
                },
            }

        return [optimizer]
