"""astro-denoise-bm3d: BM3D classical baseline for astronomical image denoising."""

__version__ = "0.1.0"

from .bm3d_denoiser import BM3DDenoiser, estimate_sigma_from_variance
from .interface import DenoiseMethod

__all__ = ["DenoiseMethod", "BM3DDenoiser", "estimate_sigma_from_variance", "__version__"]
