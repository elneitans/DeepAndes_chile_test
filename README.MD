# Geoglif Project Guide

This guide focuses on the repository-level layout. For extraction and analysis details, see [dataset/README.md](dataset/README.md).

## Data folder (how to create/populate)

- Expected shape:
	- `data/<area>_raw/`: inputs and outputs for an area (e.g., `unita_raw`, `chugchug_raw`, `lluta_raw`, `granllama_raw`, `salvador_raw`). Place the orthomosaic GeoTIFFs here; extraction scripts will write ortho JPEG/PNG crops and overlays back into the same folder.
	- `data/<area>_polygons/`: polygon sources and their derived files. Store the source geopackage/shapefile and the generated `summary.json`. When you run `dataset/extract_polygon_images.py`, it reads the geodata here and writes per-polygon metadata JSONs and any additional exports alongside the summary.
	- `data/<area>_geos/`: optional filtered copies that keep only class 1 (geo) crops/metadata for rapid browsing. `dataset/analysis.py` and helper routines can populate these when you ask to copy geo-class images.
- Typical naming: each polygon produces `geoglif_XXXX_metadata.json` plus ortho/overlay image files referenced inside the JSON. The `summary.json` inside `<area>_polygons` aggregates them.
- Minimal creation steps for a new area:
	1) Create the three folders: `<area>_raw`, `<area>_polygons`, `<area>_geos` under `data/`.
	2) Drop the orthomosaic into `<area>_raw/` and the labeled polygons (e.g., geopackage) into `<area>_polygons/`.
	3) Run the extractor from `dataset/` (see its README) pointing to those inputs; outputs stay under the same area folders.
	4) Optionally run the analysis helpers to fill `<area>_geos/` with class-1 crops for quick QA.

## Tests folder (what it is and how to run)

The [Tests](Tests) directory contains small analysis scripts that compute spatial metrics over extracted crops using the `summary.json` files in `data/`:
- [Tests/moran.py](Tests/moran.py): computes Moran’s I per crop (grayscale, resized), saves CSV plus box/hist plots.
- [Tests/entropy.py](Tests/entropy.py): computes spatial entropy per crop, saves CSV plus box/hist plots.
- [Tests/combined.py](Tests/combined.py): computes both metrics together and produces a scatter plot; configure `AREA_SUBDIR` to point at the target `data/<area>_polygons` folder and adjust `TARGET_SIZE` if needed.
- [Tests/main.py](Tests/main.py): convenience printout showing the resolved project/data paths.

Quick use:
```
cd Tests
python moran.py      # or entropy.py / combined.py
```
Make sure the corresponding `data/<area>_polygons/summary.json` and referenced images exist; update any hard-coded `AREA_SUBDIR` values before running.

## Environment and requirements

- Install dependencies once from the project root: `pip install -r requirements.txt`.
- Recommended: use a fresh virtual environment (e.g., `python -m venv .venv` then `.venv\Scripts\activate` on Windows).
- Key libraries: `rasterio`, `geopandas`, `shapely`, `pyproj`, `pysal`, `libpysal`, `esda`, `matplotlib`, `opencv-python`, `notebook`. If GDAL/PROJ issues appear, ensure system-level geospatial deps are available (via conda-forge if preferred).
- For dataset usage, follow the workflows in [dataset/README.md](dataset/README.md) after installing requirements.
