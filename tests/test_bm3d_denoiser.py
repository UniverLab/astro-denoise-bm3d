"""Unit tests for sigma estimation and the BM3D denoiser."""

import numpy as np
import pytest

from astro_denoise_bm3d.bm3d_denoiser import (
    BM3DDenoiser,
    estimate_sigma_from_variance,
    estimate_sigma_mad,
)


class TestSigmaEstimation:
    def test_from_variance_uses_median(self):
        variance = np.full((32, 32), 4.0)
        variance[0, :] = 400.0  # bright-source pixels must not dominate
        assert estimate_sigma_from_variance(variance) == pytest.approx(2.0)

    def test_from_variance_ignores_invalid(self):
        variance = np.array([[np.nan, -1.0, 0.0, 9.0, 9.0]])
        assert estimate_sigma_from_variance(variance) == pytest.approx(3.0)

    def test_from_variance_all_invalid_raises(self):
        with pytest.raises(ValueError):
            estimate_sigma_from_variance(np.zeros((4, 4)))

    def test_mad_recovers_gaussian_sigma(self):
        rng = np.random.default_rng(0)
        patch = rng.normal(100.0, 5.0, size=(256, 256))
        assert estimate_sigma_mad(patch) == pytest.approx(5.0, rel=0.05)


class TestBM3DDenoiser:
    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError):
            BM3DDenoiser(stage="wiener")

    def test_non_2d_patch_raises(self):
        with pytest.raises(ValueError):
            BM3DDenoiser(sigma=1.0).denoise(np.zeros(16))

    def test_reduces_noise_and_preserves_scale(self):
        rng = np.random.default_rng(1)
        clean = np.full((64, 64), 100.0)
        clean[24:40, 24:40] += 50.0  # extended source
        noisy = clean + rng.normal(0, 5.0, clean.shape)

        denoised = BM3DDenoiser(sigma=5.0).denoise(noisy)

        assert denoised.shape == noisy.shape
        residual_before = np.std(noisy - clean)
        residual_after = np.std(denoised - clean)
        assert residual_after < 0.5 * residual_before
        # Original scale restored (sky level preserved).
        assert np.median(denoised) == pytest.approx(np.median(clean), rel=0.05)

    def test_zero_sigma_returns_copy(self):
        patch = np.random.default_rng(2).normal(size=(32, 32))
        out = BM3DDenoiser(sigma=0.0).denoise(patch)
        np.testing.assert_array_equal(out, patch)
        assert out is not patch

    def test_set_sigma_from_variance(self):
        denoiser = BM3DDenoiser()
        sigma = denoiser.set_sigma_from_variance(np.full((8, 8), 16.0))
        assert sigma == pytest.approx(4.0)
        assert denoiser.sigma == pytest.approx(4.0)

    def test_name(self):
        assert BM3DDenoiser(sigma=1.0).name() == "bm3d"
