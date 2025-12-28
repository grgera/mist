from huggingface_hub import PyTorchModelHubMixin
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Sequence, Any

from mist_statinf.utils.format import as_2col, pad_truncate
from .estimator import MISTModel

class MISTForHF(nn.Module, PyTorchModelHubMixin):
    """
    Hugging Face-compatible wrapper for MIST mutual information estimation.
    Provides high-level methods: estimate_point() and estimate_interval_qr().
    """
    def __init__(
        self,
        # --- MISTModel args ---
        n_phi_layers: int,
        phi_hidden_dim: int,
        n_phi_heads: int,
        phi_dim_forward: int,
        phi_activation_fun: str,
        n_rho_layers: int,
        rho_hidden_dim: int,
        n_inds: int,
        n_seeds: Optional[int],
        n_dec_layers: Optional[int],
        sab_stack_layers: int,
        max_input_dim: int,
        output_dim: int = 1,
        phi_model: str = "set_transformer",
        quantile_conditioned: bool = False,

        # --- Metadata for HF ---
        repo_url: Optional[str] = None,
        license: str = "mit",
        **kwargs
    ):
        super().__init__()
        # Store config for reconstruction
        self.config = {
            "n_phi_layers": n_phi_layers,
            "phi_hidden_dim": phi_hidden_dim,
            "n_phi_heads": n_phi_heads,
            "phi_dim_forward": phi_dim_forward,
            "phi_activation_fun": phi_activation_fun,
            "n_rho_layers": n_rho_layers,
            "rho_hidden_dim": rho_hidden_dim,
            "n_inds": n_inds,
            "n_seeds": n_seeds,
            "n_dec_layers": n_dec_layers,
            "sab_stack_layers": sab_stack_layers,
            "max_input_dim": max_input_dim,
            "output_dim": output_dim,
            "phi_model": phi_model,
            "quantile_conditioned": quantile_conditioned,
        }

        # Build the core model
        self.core = MISTModel(**self.config)

        # Cache expected input dim for preprocessing
        self.target_feature_dim = self.core.phi.embedding_mlp[0].in_features

        # Register as buffer to track quantile mode (for safety)
        self.quantile_conditioned = quantile_conditioned

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.core(batch)

    # --- High-level API (exposed to users) ---

    def _prepare_source(self, X: np.ndarray, Y: np.ndarray) -> torch.Tensor:
        arr = as_2col(X, Y)
        x = torch.tensor(arr, dtype=torch.float32, device=next(self.parameters()).device).unsqueeze(0)
        x = pad_truncate(x, self.target_feature_dim)
        return x

    @torch.no_grad()
    def estimate_point(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        tau: Optional[float] = None,
    ) -> float:
        x = self._prepare_source(X, Y)
        batch = {"source": x}
        if self.quantile_conditioned:
            _t = 0.5 if tau is None else float(tau)
            batch["tau"] = torch.full((1, 1), _t, device=x.device)
        out = self.core(batch)
        return float(out.item())

    @torch.no_grad()
    def estimate_interval_qr(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        lower: float = 0.05,
        upper: float = 0.95,
        include_median: bool = True,
    ) -> Dict[str, float]:
        if not self.quantile_conditioned:
            raise RuntimeError("This model was not trained for quantile regression.")

        x = self._prepare_source(X, Y)
        taus = [float(lower), float(upper)]
        if include_median:
            taus.append(0.5)

        k = len(taus)
        x_rep = x.expand(k, -1, -1).contiguous()
        tau_tensor = torch.tensor(taus, dtype=torch.float32, device=x.device).view(-1, 1)

        batch = {"source": x_rep, "tau": tau_tensor}
        out = self.core(batch).view(-1)
        vals = out.cpu().numpy()

        result = {"lower": float(vals[0]), "upper": float(vals[1])}
        if include_median:
            result["median"] = float(vals[2])
        return result