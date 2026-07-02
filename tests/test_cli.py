"""End-to-end test of the bench CLI protocol for the BM3D module."""

import json

import numpy as np
import pytest
from astropy.io import fits

from astro_denoise_bm3d.__main__ import main


@pytest.fixture
def noisy_fits(tmp_path):
    rng = np.random.default_rng(0)
    clean = np.full((64, 64), 100.0)
    clean[20:30, 20:30] += 80.0
    noisy = clean + rng.normal(0, 4.0, clean.shape)
    path = tmp_path / "input.fits"
    fits.PrimaryHDU(noisy.astype(np.float32)).writeto(path)
    return path, clean


def test_cli_denoises_and_emits_json(noisy_fits, tmp_path, capsys):
    input_path, clean = noisy_fits
    output_path = tmp_path / "output.fits"
    rc = main(["--input", str(input_path), "--output", str(output_path)])
    assert rc == 0

    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert status["algorithm"] == "bm3d"
    assert status["status"] == "ok"
    assert status["sigma"] is None or status["sigma"] > 0

    with fits.open(output_path) as hdul:
        denoised = np.asarray(hdul[0].data, dtype=np.float64)
        history = str(hdul[0].header.get("HISTORY", ""))
    assert denoised.shape == clean.shape
    assert np.std(denoised - clean) < np.std(
        fits.getdata(input_path).astype(np.float64) - clean
    )
    assert "astro-denoise-bm3d" in history


def test_cli_config_sigma_and_variance_hdu(noisy_fits, tmp_path, capsys):
    input_path, _ = noisy_fits
    # Rewrite the input with an attached variance plane in HDU 1.
    image = fits.getdata(input_path).astype(np.float32)
    hdul = fits.HDUList([
        fits.PrimaryHDU(image),
        fits.ImageHDU(np.full(image.shape, 16.0, dtype=np.float32), name="VARIANCE"),
    ])
    stacked = tmp_path / "stacked.fits"
    hdul.writeto(stacked)

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"variance_hdu": 1}))
    rc = main([
        "--input", str(stacked),
        "--output", str(tmp_path / "out.fits"),
        "--config", str(config),
    ])
    assert rc == 0
    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert status["sigma"] == pytest.approx(4.0)


def test_cli_error_is_structured(tmp_path, capsys):
    rc = main([
        "--input", str(tmp_path / "missing.fits"),
        "--output", str(tmp_path / "out.fits"),
    ])
    assert rc == 1
    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert status["status"] == "error"
