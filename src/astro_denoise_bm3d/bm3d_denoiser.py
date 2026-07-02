"""BM3D denoiser conforming to the astro-denoise DenoiseMethod contract.

Astronomical patches have arbitrary dynamic range, while BM3D assumes
data roughly in [0, 1]. The denoiser therefore normalizes each patch
with robust (median/percentile) statistics, runs BM3D with the noise
sigma expressed in normalized units, and restores the original scale.

The noise level can be given explicitly, estimated from the DC2
variance plane (``estimate_sigma_from_variance``), or estimated from
the patch itself via the median absolute deviation.
"""

from __future__ import annotations

import bm3d as _bm3d
import numpy as np

from .interface import DenoiseMethod


def estimate_sigma_from_variance(variance: np.ndarray) -> float:
    """Per-patch noise sigma from a variance plane (DC2 stores one per coadd).

    Uses the median of the variance plane, which is robust against the
    higher variance recorded on bright-source pixels.
    """
    variance = np.asarray(variance, dtype=np.float64)
    valid = variance[np.isfinite(variance) & (variance > 0)]
    if valid.size == 0:
        raise ValueError("variance plane has no positive finite values")
    return float(np.sqrt(np.median(valid)))


def estimate_sigma_mad(patch: np.ndarray) -> float:
    """Fallback noise estimate: scaled MAD of the patch background."""
    patch = np.asarray(patch, dtype=np.float64)
    finite = patch[np.isfinite(patch)]
    mad = np.median(np.abs(finite - np.median(finite)))
    return float(1.4826 * mad)


class BM3DDenoiser(DenoiseMethod):
    """Block-Matching 3D collaborative filtering baseline.

    Parameters
    ----------
    sigma : float, optional
        Noise standard deviation in image units. If omitted, it is
        estimated per patch with the scaled MAD.
    stage : {"all", "hard"}
        ``"all"`` runs both BM3D stages (hard thresholding + Wiener),
        ``"hard"`` only the first stage.
    """

    def __init__(self, sigma: float | None = None, stage: str = "all"):
        if stage not in ("all", "hard"):
            raise ValueError(f"stage must be 'all' or 'hard', got {stage!r}")
        self.sigma = sigma
        self._stage = (
            _bm3d.BM3DStages.ALL_STAGES if stage == "all" else _bm3d.BM3DStages.HARD_THRESHOLDING
        )

    def name(self) -> str:
        return "bm3d"

    def set_sigma_from_variance(self, variance: np.ndarray) -> float:
        self.sigma = estimate_sigma_from_variance(variance)
        return self.sigma

    def denoise(self, patch: np.ndarray) -> np.ndarray:
        patch = np.asarray(patch, dtype=np.float64)
        if patch.ndim != 2:
            raise ValueError(f"expected a 2-D patch, got shape {patch.shape}")

        sigma = self.sigma if self.sigma is not None else estimate_sigma_mad(patch)
        if not np.isfinite(sigma) or sigma <= 0:
            return patch.copy()

        # Robust normalization to a ~[0, 1] working range for BM3D.
        finite = patch[np.isfinite(patch)]
        offset = float(np.median(finite))
        scale = float(np.percentile(finite, 99.5) - offset)
        scale = max(scale, 10.0 * sigma)  # keep low-contrast patches stable

        normalized = (np.nan_to_num(patch, nan=offset) - offset) / scale
        denoised = _bm3d.bm3d(normalized, sigma_psd=sigma / scale, stage_arg=self._stage)
        return np.asarray(denoised, dtype=np.float64) * scale + offset
