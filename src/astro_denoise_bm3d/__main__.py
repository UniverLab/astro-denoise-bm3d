"""CLI: denoise a FITS image with BM3D following the bench protocol.

Usage:
    python -m astro_denoise_bm3d --input input.fits --output output.fits \
        [--config config.json]

stdout emits a single JSON line, e.g.:
    {"algorithm": "bm3d", "version": "0.1.0", "status": "ok"}

Config keys (all optional):
    sigma          explicit noise sigma in image units
    stage          "all" (default) or "hard"
    variance_hdu   HDU index/name of the variance plane used to
                   estimate sigma (DC2 coadds store it in HDU 3)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from rich.console import Console

from . import __version__
from .bm3d_denoiser import BM3DDenoiser


def load_config(path):
    if path is None:
        return {}
    return json.loads(Path(path).read_text())


def find_image_hdu(hdul):
    for index, hdu in enumerate(hdul):
        if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
            return index
    raise ValueError("no 2-D image HDU found in input FITS")


def emit(payload):
    print(json.dumps(payload))


def main(argv=None):
    console = Console(stderr=True)
    parser = argparse.ArgumentParser(
        prog="python -m astro_denoise_bm3d",
        description="BM3D denoising baseline for astronomical FITS images.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        denoiser = BM3DDenoiser(sigma=config.get("sigma"), stage=config.get("stage", "all"))

        with fits.open(args.input) as hdul:
            image_hdu = find_image_hdu(hdul)
            image = np.asarray(hdul[image_hdu].data, dtype=np.float64)
            header = hdul[image_hdu].header.copy()

            variance_hdu = config.get("variance_hdu")
            if denoiser.sigma is None and variance_hdu is not None:
                variance = np.asarray(hdul[variance_hdu].data, dtype=np.float64)
                denoiser.set_sigma_from_variance(variance)
                console.print(f"sigma from variance plane: {denoiser.sigma:.4g}")

        start = time.perf_counter()
        denoised = denoiser.denoise(image)
        elapsed = time.perf_counter() - start

        header["HISTORY"] = f"astro-denoise-bm3d {__version__}"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(data=denoised.astype(np.float32), header=header).writeto(
            args.output, overwrite=True
        )
        console.print(f"[green]ok[/green] {args.input.name} -> {args.output.name} ({elapsed:.2f}s)")
        emit(
            {
                "algorithm": "bm3d",
                "version": __version__,
                "status": "ok",
                "input": str(args.input),
                "output": str(args.output),
                "sigma": denoiser.sigma,
                "elapsed_s": round(elapsed, 4),
            }
        )
        return 0
    except Exception as exc:
        console.print(f"[red]error:[/red] {exc}")
        emit({"algorithm": "bm3d", "version": __version__, "status": "error", "error": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
