# astro-denoise-bm3d

> Baseline clásico de referencia: BM3D (C-optimizado) para el benchmark
> reproducible de denoising en imágenes astronómicas DC2/LSST.

## Rol y responsabilidades

- Aplica BM3D (paquete pip `bm3d`, versión rápida C-optimizada) a parches o
  imágenes completas FITS del DC2.
- Estima el sigma de ruido desde el plano de varianzas (en los coadds DC2 es
  el HDU 3), desde config explícito, o con MAD sobre el propio patch.
- Implementa la interfaz estándar `DenoiseMethod` y el protocolo CLI del
  orquestador (`astro-denoise-bench`), con salida JSON por stdout.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # registra el entry point astro_denoise.methods/bm3d
```

## API

```python
from astro_denoise_bm3d import BM3DDenoiser, estimate_sigma_from_variance

denoiser = BM3DDenoiser(sigma=19.4)          # sigma explícito, o
denoiser.set_sigma_from_variance(var_plane)  # desde plano de varianzas
clean = denoiser.denoise(patch)              # ndarray 2-D -> ndarray 2-D
```

`BM3DDenoiser` normaliza cada patch con estadísticos robustos
(mediana / percentil 99.5) antes de aplicar BM3D y restaura la escala
original, de modo que funciona con el rango dinámico arbitrario de las
imágenes astronómicas.

## CLI (protocolo bench)

```bash
python -m astro_denoise_bm3d \
    --input input.fits \
    --output output.fits \
    --config config.json
```

`config.json` (todas las claves opcionales):

```json
{
  "sigma": 19.4,
  "stage": "all",
  "variance_hdu": 3
}
```

stdout devuelve una línea JSON con los metadatos mínimos del protocolo:

```json
{"algorithm": "bm3d", "version": "0.1.0", "status": "ok"}
```

## Notas

- No usar la versión slow/pura-Python de BM3D: este baseline usa la rápida y
  C-optimizada del paquete `bm3d`.
- El módulo se descubre automáticamente vía el entry point
  `astro_denoise.methods` cuando se instala con `pip install -e .`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Cubren la estimación de sigma (variance plane y MAD), la reducción efectiva
de ruido del denoiser y el protocolo CLI end-to-end con FITS temporales.

---
Este submódulo forma parte de la plataforma `astro-denoise`; véase
[`CONTEXT_v2.md`](../../CONTEXT_v2.md) para arquitectura y motivación.
