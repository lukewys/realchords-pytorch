"""MLX inference models for ReaLJam."""

MLX_IMPORT_ERROR = None

try:
    from realchords.realjam.mlx_models.online_model import MLXDecoderTransformer
    from realchords.realjam.mlx_models.offline_model import (
        MLXEncoderDecoderTransformer,
    )
except ImportError as exc:
    MLX_IMPORT_ERROR = exc
    MLXDecoderTransformer = None
    MLXEncoderDecoderTransformer = None

__all__ = [
    "MLXDecoderTransformer",
    "MLXEncoderDecoderTransformer",
    "MLX_IMPORT_ERROR",
]
