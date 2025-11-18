"""Template class for Transformer models with all the tricks."""

import x_transformers
import copy

DEFAULT_KWARGS = {
    # pre-normalization: x-transformers has default pre-normalization
    # ff_multi: x-transformers has default ff dim 4x of hidden dim
    "rotary_pos_emb": True,  # https://arxiv.org/abs/2104.09864
    "use_simple_rmsnorm": True,  # https://arxiv.org/abs/2307.14995
    "attn_qk_norm": True,  # https://arxiv.org/abs/2010.04245
    "attn_qk_norm_dim_scale": True,  # https://arxiv.org/abs/2302.05442
    # swiglu: https://arxiv.org/abs/2002.05202
    "ff_swish": True,
    "ff_glu": True,
    # Flash attention: https://arxiv.org/abs/2205.14135
    # Need to build flash attention to enable it.
    # https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
    "attn_flash": True,
    # T5 relative position bias cannot be used with flash attention
    # "rel_pos_bias": True,  # T5 relative position bias
}


def Decoder(
    dim: int,
    depth: int,
    heads: int,
    dropout: float = 0.1,
    cross_attend: bool = True,
    attention_layer_configs: dict = dict(),
):
    attention_layer_kwargs = copy.deepcopy(DEFAULT_KWARGS)
    attention_layer_kwargs.update(attention_layer_configs)
    return x_transformers.Decoder(
        dim=dim,
        depth=depth,
        heads=heads,
        attn_dropout=dropout,
        ff_dropout=dropout,
        cross_attend=cross_attend,
        **attention_layer_kwargs,
    )


def Encoder(
    dim: int,
    depth: int,
    heads: int,
    dropout: float = 0.1,
    attention_layer_configs: dict = dict(),
):
    attention_layer_kwargs = copy.deepcopy(DEFAULT_KWARGS)
    attention_layer_kwargs.update(attention_layer_configs)
    return x_transformers.Encoder(
        dim=dim,
        depth=depth,
        heads=heads,
        attn_dropout=dropout,
        ff_dropout=dropout,
        **attention_layer_kwargs,
    )
