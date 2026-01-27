# Geoglyph Image Analysis Test Suite

Test suite for analyzing geoglyph images using spatial and textural metrics. Processes grayscale images and computes statistical measures to characterize spatial patterns and texture properties.

## Quick Start

```bash
python repr_test.py --area lluta --methods moran entropy contrast homogeneity --output ./results
```

## Usage

### Command Structure

```bash
python repr_test.py --area <area_name> --methods <method1> [method2 ...] --output <output_dir>
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--area` | Yes | - | Area name (e.g., 'lluta', 'chugchug', 'unita') |
| `--methods` | Yes | - | Space-separated methods to compute |
| `--output` | No | `./results` | Output directory for results |
| `--size` | No | `100 100` | Target image size as width height |

### Available Methods

- `moran` - Moran's I spatial autocorrelation
- `entropy` - Spatial entropy from neighbor pairs
- `contrast` - GLCM contrast (local variations)
- `homogeneity` - GLCM homogeneity (texture uniformity)

### Examples

**Single method:**
```bash
python repr_test.py --area lluta --methods moran --output ./results
```

**Multiple methods:**
```bash
python repr_test.py --area chugchug --methods moran entropy contrast --output ./results
```

**All metrics with custom size:**
```bash
python repr_test.py --area unita --methods moran entropy contrast homogeneity --output ./output --size 150 150
```

## Data Folder Structure

The script expects data organized as follows:

```
../data/
  {area_name}_polygons/
    summary.json           # Required metadata file
    image1.jpg             # Geoglyph images
    image2.jpg
    ...
```

### Required: summary.json Format

The `summary.json` file must contain polygon metadata:

```json
{
  "polygons": [
    {
      "polygon_index": 0,
      "class": "figurative",
      "files": {
        "ortho_jpeg": "geoglyph_0000.jpg"
      },
      "bbox_size_meters": [10.5, 8.3]
    },
    {
      "polygon_index": 1,
      "class": "geometric",
      "files": {
        "ortho_jpeg": "geoglyph_0001.jpg"
      },
      "bbox_size_meters": [12.0, 15.2]
    }
  ]
}
```

**Key fields:**
- `polygon_index` - Unique identifier for the geoglyph
- `class` - Classification label (e.g., "figurative", "geometric", "anthropomorphic")
- `files.ortho_jpeg` - Filename of the image (must exist in same directory)
- `bbox_size_meters` - Real-world dimensions (for reference only)

## Output Files

Results are saved to the output directory:

**CSV:**
- `{area_name}_results.csv` - Computed metrics for each image

**Visualizations:**
- `{area_name}_{method}_histogram.png` - Distribution histograms
- `{area_name}_{method}_boxplot.png` - Boxplots by class
- `{area_name}_{method1}_vs_{method2}_scatter.png` - Pairwise scatter plots
- `{area_name}_3d_scatter.png` - 3D scatter (if 3+ metrics computed)

## Example Workflow

```bash
# 1. Ensure data folder structure is correct
ls ../data/lluta_polygons/
# Should show: summary.json and *.jpg files

# 2. Run analysis
python repr_test.py --area lluta --methods moran entropy contrast homogeneity --output ./lluta_results

# 3. View results
cat ./lluta_results/lluta_results.csv
```

## Dependencies

Install required packages:
```bash
pip install -r ../requirements.txt
```

Main dependencies: numpy, pandas, Pillow, scikit-image, libpysal, esda, matplotlib, seaborn

## Common Issues

**"Summary not found" error:**
- Verify `../data/{area_name}_polygons/summary.json` exists
- Check that area name matches folder name exactly

**Image loading errors:**
- Ensure all images referenced in summary.json exist
- Verify image filenames match exactly (case-sensitive)