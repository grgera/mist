from __future__ import annotations
from typing import Optional, Dict, Sequence, Union
import numpy as np
import torch

from .train.lit_module import MISTModelLit
from .model.estimator import MISTModel
from .utils import resolve_device, as_2col, pad_truncate, load_quickstart_cfg

Number = Union[float, np.floating]

class MISTQuickEstimator:
    """
    A wrapper class for fast inference with MIST or MIST_QR.

    Args:
        loss: "mse" (point estimate) or "qr" (QCQR quantile model).
        checkpoint: path to ckpt/pt file. 
        device: "auto" | "cpu" | "cuda" | "mps".
        config_path: YAML file defining the architecture (as used during training).
    """
    def __init__(
        self,
        loss: str = "mse",
        checkpoint: Optional[str] = None,
        device: str = "auto",
        config_path: Optional[str] = None,
    ):
        self.device = resolve_device(device)
        loss = loss.lower()
        if loss not in {"mse", "qr"}:
            raise ValueError(f"Unknown loss type {loss}, expected 'mse' or 'qr'")
        self.loss_type = "MSE" if loss == "mse" else "QCQR"

        # --- load architecture from YAML ---
        cfg_dict = load_quickstart_cfg(config_path)
        arch = dict(cfg_dict.get("architecture", {}))
        opt = dict(cfg_dict.get("optimizer", {})) 

        arch["quantile_conditioned"] = (self.loss_type == "QCQR")
        arch.setdefault("output_dim", 1)
        arch.setdefault("phi_model", "set_transformer")

        # --- Load model ---
        if checkpoint and checkpoint.endswith(".ckpt"):
            self.model_lit = MISTModelLit.load_from_checkpoint(
                checkpoint,
                args={
                    "loss_type": self.loss_type,
                    "architecture": arch,
                    "optimizer": opt,
                    "trainer": {"max_epochs": 1},
                },
            )
        else:
            core = MISTModel(**arch)
            if checkpoint:
                state = torch.load(checkpoint, map_location="cpu")
                state = state.get("state_dict", state)
                fixed = {}
                for k, v in state.items():
                    if k.startswith("msm."):
                        fixed[k[len("msm."):]] = v
                if fixed:
                    core.load_state_dict(fixed, strict=False)
                else:
                    core.load_state_dict(state, strict=False)

            class _Lite(torch.nn.Module):
                def __init__(self, msm, loss_type):
                    super().__init__()
                    self.msm = msm
                    self.loss_type = loss_type

            self.model_lit = _Lite(core, self.loss_type)

        self.model_lit.to(self.device).eval()

        self.target_feature_dim = (
            self.model_lit.msm.phi.embedding_mlp[0].in_features
        )

    # ---------- Helpers ----------

    def _prepare_source(self, X: np.ndarray, Y: np.ndarray) -> torch.Tensor:
        """
        Converts (X, Y) into a tensor of shape (1, N, D), including
        padding/truncation to match the expected feature dimension.
        """
        arr = as_2col(X, Y)  # (N, D)
        x = torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        x = pad_truncate(x, self.target_feature_dim)
        return x  # (1, N, D)

    @torch.no_grad()
    def _forward_single(
        self,
        x: torch.Tensor,
        tau: Optional[float] = None,
    ) -> float:
        """
        Performs a single forward pass of the model.

          - MSE mode:   msm({"source": x})
          - QCQR mode:  msm({"source": x, "tau": ...})
        """
        batch = {"source": x}
        if self.loss_type == "QCQR":
            _t = 0.5 if tau is None else float(tau)
            batch["tau"] = torch.full((1, 1), _t, device=self.device)
        out = self.model_lit.msm(batch)
        return float(out.item())

    @torch.no_grad()
    def _forward_many_taus(
        self,
        x: torch.Tensor,
        taus: Sequence[float],
    ) -> Dict[float, float]:
        """
        Vectorized QCQR forward pass for multiple τ values in a single call.
        """
        if self.loss_type != "QCQR":
            raise RuntimeError("forward_many_taus is only valid for QCQR models")

        taus = [float(t) for t in taus]
        k = len(taus)

        # Repeat the source tensor k times along the batch dimension:
        # (1, N, D) -> (k, N, D)
        x_rep = x.expand(k, -1, -1).contiguous()

        tau_tensor = torch.tensor(taus, dtype=torch.float32, device=self.device).view(-1, 1)

        batch = {
            "source": x_rep,
            "tau": tau_tensor,
        }
        out = self.model_lit.msm(batch).view(-1)  # (k,)
        vals = out.detach().cpu().numpy().tolist()
        return {t: float(v) for t, v in zip(taus, vals)}

    # ---------- Public API ----------

    @torch.no_grad()
    def estimate_point(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        tau: Optional[float] = None,
    ) -> Number:
        """
        Computes a point estimate of mutual information.

        MSE model:
            - `tau` is ignored → returns a single scalar.
        QCQR model:
            - if tau=None → returns the median estimate (τ = 0.5),
            - otherwise → returns the τ-quantile estimate.
        """
        x = self._prepare_source(X, Y)
        return self._forward_single(x, tau=tau)

    @torch.no_grad()
    def estimate_interval_qr(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        lower: float = 0.05,
        upper: float = 0.95,
        include_median: bool = True,
    ) -> Dict[str, float]:
        """
        Computes a quantile-based uncertainty interval for a QCQR model
        without relying on bootstrap sampling.

        Returns a dictionary:
            {
              "lower":  q_lower,
              "upper":  q_upper,
              "median": q_med    # only if include_median=True
            }
        """
        if self.loss_type != "QCQR":
            raise RuntimeError("estimate_interval_qr is only valid for QCQR models")

        x = self._prepare_source(X, Y)

        taus = [lower, upper]
        if include_median:
            taus.append(0.5)
        vals = self._forward_many_taus(x, taus)

        result = {
            "lower": vals[float(lower)],
            "upper": vals[float(upper)],
        }
        if include_median:
            result["median"] = vals[0.5]
        return result
