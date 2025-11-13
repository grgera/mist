import torch
import torch.nn as nn
from typing import Dict

class FeatureInteractionBlock(nn.Module):
    """
    A stacked Set-Attention block: [SAB → LayerNorm → Dropout] × n_layers.

    Args:
        n_layers (int): Number of SAB–norm–dropout blocks to stack.
        embed_dim (int): Token embedding dimension.
        num_heads (int): Number of attention heads in SAB.
        batch_first (bool): If True, tensors use (B, L, D) layout; otherwise (L, B, D).
        activation (str): Feed-forward activation inside SAB ("gelu" or "relu").
        dropout_prob (float, optional): Dropout after each SAB+Norm. Defaults to 0.3.

    Returns:
        Tensor: Transformed features with the same shape/layout as `x`.
    """
    def __init__(self, n_layers, embed_dim, num_heads, batch_first, activation, dropout_prob=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(SAB(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=batch_first,
                activation=activation
            ))
            self.layers.append(nn.LayerNorm(embed_dim))
            self.layers.append(nn.Dropout(dropout_prob))
    
    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            if isinstance(layer, SAB):
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
            else:
                x = layer(x)
        return x

class MAB(nn.Module):
    """
    Multihead-Attention Block (Set Transformer component).

    Structure: x ← LN(x + MHA(x, y)) → h;  output ← LN(h + FF(h))

    Args:
        embed_dim (int): Model/embedding dimension (D).
        dim_feedforward (int): Hidden size of the feed-forward sublayer.
        num_heads (int): Number of attention heads.
        batch_first (bool): If True, tensors use (B, L, D); else (L, B, D).
        activation (str): Feed-forward activation ("gelu"/"Gelu" or "relu").

    Returns:
        Tensor: Output tensor with the same shape as `x`.
    """
    def __init__(self, embed_dim, dim_feedforward, num_heads, batch_first, activation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=0, batch_first=batch_first)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        if activation == "gelu" or activation == "Gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def _ff_block(self, x):
        return self.linear2(self.activation(self.linear1(x)))

    def forward(self, x, y, src_key_padding_mask=None, attn_mask=None):
        h = self.norm1(x + self.mha(x, y, y,
                                    attn_mask=attn_mask,
                                    key_padding_mask=src_key_padding_mask)[0])
        return self.norm2(h + self._ff_block(h))


class SAB(nn.Module):
    """
    Self-Attention Block (Set Transformer): MAB with x as both query and key/value.

    Args:
        embed_dim (int): Token/feature dimension (D).
        num_heads (int): Number of attention heads.
        batch_first (bool): If True, tensors use (B, L, D); else (L, B, D).
        activation (str): Feed-forward activation inside MAB ("gelu"/"relu").

    Returns:
        Tensor: Output with the same shape/layout as `x`.
    """
    def __init__(self, embed_dim, num_heads, batch_first, activation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mab = MAB(embed_dim=embed_dim, dim_feedforward=embed_dim, num_heads=num_heads,
                       batch_first=batch_first, activation=activation)

    def forward(self, x, src_key_padding_mask=None, attn_mask=None):
        return self.mab(x, x, src_key_padding_mask=src_key_padding_mask, attn_mask=attn_mask)


class PMA(nn.Module):
    """
    Pooling by Multihead Attention (PMA) from Set Transformer.

    Uses a learned set of seed vectors `s` to aggregate an input set `x` via MAB(s, x).

    Args:
        embed_dim (int): Embedding dimension (D).
        num_heads (int): Number of attention heads.
        num_seeds (int): Number of learned seed vectors (controls pooled set size).
        batch_first (bool): If True, tensors use (B, L, D) layout; else (L, B, D).
        activation (str): Feed-forward activation inside the internal MAB ("gelu" or "relu").

    Returns:
        Tensor: Pooled set of shape (B, num_seeds, D) (or (num_seeds, B, D) if not batch_first).
    """
    def __init__(self, embed_dim, num_heads, num_seeds, batch_first, activation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s = nn.Parameter(torch.Tensor(1, num_seeds, embed_dim))
        nn.init.xavier_uniform_(self.s)
        self.mab = MAB(embed_dim=embed_dim, dim_feedforward=embed_dim, num_heads=num_heads,
                       batch_first=batch_first, activation=activation)

    def forward(self, x, src_key_padding_mask=None, attn_mask=None):
        s_repeat = self.s.repeat(x.size(0), 1, 1)
        return self.mab(s_repeat, x, src_key_padding_mask=src_key_padding_mask, attn_mask=attn_mask)


class SetTransformerBlock(nn.Sequential):
    """
    Sequential container for Set Transformer modules (e.g., SAB/ISAB/PMA stacks).

    Forward simply passes `inputs` (and optional `src_key_padding_mask`) through each submodule.

    Returns:
        Tensor: Output after applying all submodules in sequence.
    """
    def forward(self, inputs, src_key_padding_mask=None):
        for module in self._modules.values():
            inputs = module(inputs, src_key_padding_mask=src_key_padding_mask)
        return inputs


class ISABEncoderLayer(nn.Module):
    """
    Induced Self-Attention Block (ISAB) encoder layer.

    Implements two MABs with a trainable set of inducing points `I`:
        h = MAB(I, X);  Y = MAB(X, h)

    Args:
        embed_dim (int): Embedding dimension (D).
        dim_feedforward (int): Hidden size for the MAB feed-forward sublayers.
        num_heads (int): Number of attention heads.
        batch_first (bool): If True, tensors use (B, L, D); else (L, B, D).
        num_inds (int): Number of inducing points (rows in I).
        activation (str): Feed-forward activation in MAB ("gelu" or "relu").

    Returns:
        Tensor: Output tensor with the same layout/shape as `src`.
    """
    def __init__(self, embed_dim, dim_feedforward, num_heads, batch_first, num_inds, activation, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.i = nn.Parameter(torch.Tensor(1, num_inds, embed_dim))
        nn.init.xavier_uniform_(self.i)

        self.mab0 = MAB(embed_dim=embed_dim, dim_feedforward=dim_feedforward, num_heads=num_heads,
                        batch_first=batch_first, activation=activation)

        self.mab1 = MAB(embed_dim=embed_dim, dim_feedforward=dim_feedforward, num_heads=num_heads,
                        batch_first=batch_first, activation=activation)

    def forward(
            self,
            src,
            src_mask=None,
            src_key_padding_mask=None,
    ):
        i_repeat = self.i.repeat(src.size(0), 1, 1)
        h = self.mab0(i_repeat, src, src_key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        return self.mab1(src, h)

class MISTSetTransformer(nn.Module):
    """
    MIST backbone based on Set Transformer.

    Pipeline:
        1) Embedding MLP over raw inputs (per token).
        2) FeatureInteractionBlock: stacked SAB→LN→Dropout layers.
        3) Encoder: stack of ISABEncoderLayer (induced self-attention).
        4) Pooling: PMA with `n_seeds` seed vectors to aggregate variable-length sets.
        5) Decoder: SAB stack + final PMA( num_seeds=1 ) to produce a single set representation.
        6) Output: linear head → `output_dim`.

    Args:
        output_dim (int): Output dimension (e.g., 1 for scalar MI/quantile).
        n_heads (int): Number of attention heads for SAB/ISAB/MAB.
        n_inds (int): Number of inducing points in ISAB.
        n_enc_layers (int): Number of ISAB encoder layers.
        n_dec_layers (int): Number of decoder SAB layers (before final PMA).
        dim_hidden (int): Hidden dimension D for token embeddings/attention blocks.
        dim_feedforward (int): FFN hidden size inside MAB/SAB layers.
        activation_fun (str): Activation in attention blocks ("gelu" or "relu").
        n_seeds (int): Number of PMA seeds for the first pooling stage.
        n_feature_sab_layers (int): Number of SAB layers in the feature interaction stack.
        max_input_dim (int): Maximum per-token input feature dimension.
        dropout (float, optional): Dropout used in the embedding MLP. Defaults to 0.1.

    Returns:
        Tensor: Pooled representation passed through a linear head, shape (B, output_dim).
    """
    def __init__(
            self,
            output_dim: int,
            n_heads: int,
            n_inds: int,
            n_enc_layers: int,
            n_dec_layers: int,
            dim_hidden: int,
            dim_feedforward: int,
            activation_fun: str,
            n_seeds: int,
            n_feature_sab_layers: int,  # Number of SAB in stack
            max_input_dim: int,
            dropout=0.1
    ):
        super().__init__()

        self.embedding_mlp = nn.Sequential(
            nn.Linear(max_input_dim, 2*dim_hidden),
            nn.GELU(),
            nn.LayerNorm(2*dim_hidden),
            nn.Dropout(dropout),
            nn.Linear(2*dim_hidden, dim_hidden),
            nn.GELU(),
            nn.LayerNorm(dim_hidden)
        )

        self.feature_interaction_block = FeatureInteractionBlock(
            n_layers=n_feature_sab_layers,  
            embed_dim=dim_hidden,
            num_heads=n_heads,
            batch_first=True,
            activation=activation_fun,
            dropout_prob=0.3  
        )
        
        enc_layers = []
        for _ in range(n_enc_layers):
            enc_layers.append(ISABEncoderLayer(
                embed_dim=dim_hidden,
                num_heads=n_heads,
                num_inds=n_inds,
                dim_feedforward=dim_feedforward,
                activation=activation_fun,
                batch_first=True,
            ))
        self.encoder = SetTransformerBlock(*enc_layers)

        self.pooling_pma = PMA(
            embed_dim=dim_hidden,
            num_heads=n_heads,
            num_seeds=n_seeds,
            batch_first=True,
            activation=activation_fun
        )

        dec_layers = []
        for _ in range(n_dec_layers):
            dec_layers.append(SAB(embed_dim=dim_hidden, num_heads=n_heads, batch_first=True, activation=activation_fun))
        dec_layers.append(
            PMA(embed_dim=dim_hidden, num_heads=n_heads, num_seeds=1, batch_first=True, activation=activation_fun)
        )
        self.decoder = SetTransformerBlock(*dec_layers)
        self.output_layer = nn.Linear(dim_hidden, output_dim)
        
    def forward(self, batch: Dict):
        x = batch["source"]  # (batch_size, seq_len, input_dim)
        padding_mask_seq = batch.get("padding_mask", None) 
    
        padding_mask_input = (x.sum(dim=-1) == 0)  # (batch_size, seq_len)
    
        x = self.embedding_mlp(x)

        x = self.feature_interaction_block(x, src_key_padding_mask=padding_mask_input)
        
        x = self.encoder(x, src_key_padding_mask=padding_mask_seq)
        x = self.pooling_pma(x, src_key_padding_mask=padding_mask_seq)
        x = self.decoder(x)
        x = x[:, 0, :]
        
        return self.output_layer(x)