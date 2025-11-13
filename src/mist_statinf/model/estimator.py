
from typing import Dict
import torch
import torch.nn as nn

from .architecture import MISTSetTransformer


class MISTModel(nn.Module):
    """
    Transformer-based MI estimator with φ/ρ decomposition and optional quantile conditioning.

    The encoder φ is a Set Transformer (`MISTSetTransformer`) that maps a variable-length
    set of samples to a fixed-length representation. The head ρ is an MLP that converts
    this representation to the final output. When `quantile_conditioned=True`, a given
    quantile level τ is concatenated to the φ representation before ρ.

    Args:
        n_phi_layers (int): Number of encoder (Set Transformer) layers.
        phi_hidden_dim (int): Hidden dimension produced by φ; also the input dim to ρ.
        n_phi_heads (int): Number of attention heads in φ.
        phi_dim_forward (int): FFN hidden size inside attention blocks of φ.
        phi_activation_fun (str): Activation used in φ blocks ("gelu" or "relu").
        n_rho_layers (int): Number of MLP layers in ρ (>=1). If 1, uses a single Linear.
        rho_hidden_dim (int): Hidden size of intermediate layers in ρ (ignored if `n_rho_layers==1`).
        n_inds (int): Number of inducing points for ISAB layers in φ.
        n_seeds (int | None): Number of PMA seeds in φ pooling. If None, defaults to `n_inds`.
        n_dec_layers (int | None): Number of SAB layers in φ decoder. If None, defaults to `n_phi_layers`.
        sab_stack_layers (int): Number of SAB layers in the initial feature-interaction stack inside φ.
        max_input_dim (int): Maximum per-token input feature dimension expected by φ.
        output_dim (int, optional): Output size of ρ (e.g., 1 for scalar MI/quantile). Defaults to 1.
        phi_model (str, optional): Encoder type. Only "set_transformer" is supported. Defaults to "set_transformer".
        quantile_conditioned (bool, optional): If True, concatenates τ (shape (B,1)) to φ output before ρ.

    Forward Args:
        batch (Dict): Batch dictionary expected by `MISTSetTransformer`, typically containing:
            - "source" (FloatTensor): (B, L, max_input_dim), zero-padded if needed.
            - "padding_mask" (BoolTensor, optional): (B, L), True = padded (ignored by attention).
            - "tau" (FloatTensor, optional): (B, 1) quantile level, required if `quantile_conditioned=True`.
            - other keys may be present (e.g., "target") but are not used by `forward`.

    Returns:
        torch.Tensor: Model outputs of shape (B, output_dim).

    Notes:
        - If `quantile_conditioned=True` and no `"tau"` is provided in `batch`, a `ValueError` is raised.
        - `n_seeds` and `n_dec_layers` default to `n_inds` and `n_phi_layers` respectively when not provided.
        - The encoder φ returns a (B, phi_hidden_dim) representation; τ (if provided) is concatenated along dim=1.
    """
    def __init__(
        self,
        n_phi_layers: int,
        phi_hidden_dim: int,
        n_phi_heads: int,
        phi_dim_forward: int,
        phi_activation_fun: str,
        n_rho_layers: int,
        rho_hidden_dim: int,
        n_inds: int,
        n_seeds: int | None,
        n_dec_layers: int | None,
        sab_stack_layers: int,
        max_input_dim: int,
        output_dim: int = 1,
        phi_model: str = "set_transformer",
        quantile_conditioned: bool = False,
    ):
        super().__init__()

        if phi_model not in ["set_transformer"]:
            raise ValueError(f"Unsupported encoder: {phi_model}")

        # ---- φ (SetTransformer) ------------------------------------------------
        assert n_inds is not None, "SetTransformer needs to know n_inds"
        n_seeds = n_seeds if n_seeds else n_inds
        n_dec_layers = n_dec_layers if n_dec_layers else n_phi_layers

        self.phi = MISTSetTransformer(
            output_dim=phi_hidden_dim,      
            n_heads=n_phi_heads,
            n_inds=n_inds,
            n_seeds=n_seeds,
            n_enc_layers=n_phi_layers,
            n_dec_layers=n_dec_layers,
            dim_hidden=phi_hidden_dim,
            dim_feedforward=phi_dim_forward,
            activation_fun=phi_activation_fun,
            n_feature_sab_layers=sab_stack_layers,
            max_input_dim=max_input_dim,
        )

        # ---- ρ (MLP with oprinal τ) ----------------------
        self.quantile_conditioned = quantile_conditioned
        rho_input_dim = phi_hidden_dim + (1 if self.quantile_conditioned else 0)

        layers: list[nn.Module] = []
        if n_rho_layers == 1:
            layers.append(nn.Linear(rho_input_dim, output_dim))
        else:
            layers.append(nn.Linear(rho_input_dim, rho_hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(n_rho_layers - 2):
                layers.append(nn.Linear(rho_hidden_dim, rho_hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(rho_hidden_dim, output_dim))

        self.rho = nn.Sequential(*layers)

    def forward(self, batch: Dict) -> torch.Tensor:
        dataset_repr = self.phi(batch)  # (B, phi_hidden_dim)

        if self.quantile_conditioned:
            tau = batch.get("tau", None)
            if tau is None:
                raise ValueError("quantile_conditioned=True but no 'tau' provided in batch")
            if tau.ndim == 1:
                tau = tau.unsqueeze(1)
            dataset_repr = torch.cat([dataset_repr, tau], dim=1)

        return self.rho(dataset_repr)


# ---------------------------- utils: save/load ---------------------------------

def save_estimator(model: nn.Module, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_estimator(
    weights_path: str,
    *,
    model_kwargs: Dict,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> MISTModel:
    model = MISTModel(**model_kwargs)
    state = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state, strict=strict)
    return model
