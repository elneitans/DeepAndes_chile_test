from skimage.util import view_as_windows
from pathlib import Path
import numpy as np
from PIL import Image

from utils import (
    load_tif_image,
    load_polygon_from_metadata,
    load_metadata,
    convert_polygon_geo_to_pixel,
    calculate_patch_overlap
)


WINDOW_SIZE = 224
STRIDE = 122
CHANNELS = 3
THRESHOLD = 0.8

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "resized_geos"
OUTPUT_DIR = BASE_DIR / "data" / "patches"


def extract_patches(img):
    """Extract patches from image. Returns None if too small."""
    h, w = img.shape[:2]
    if h < WINDOW_SIZE or w < WINDOW_SIZE:
        return None
    return view_as_windows(img, (WINDOW_SIZE, WINDOW_SIZE, CHANNELS), step=STRIDE)


def save_patches(patches, polygon_coords, output_dir, threshold):
    """Save patches, filtering by polygon overlap. Returns (saved_count, skipped_count)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = output_dir / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    
    saved = skipped = 0
    rows, cols = patches.shape[0], patches.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            patch = patches[i, j, 0]
            patch_name = f"patch_{i:03d}_{j:03d}.png"
            
            window = (j * STRIDE, i * STRIDE, j * STRIDE + WINDOW_SIZE, i * STRIDE + WINDOW_SIZE)
            
            path = (output_dir if calculate_patch_overlap(window, polygon_coords, threshold) else rejected_dir) / patch_name
            Image.fromarray(patch.astype(np.uint8)).save(path)
            
            saved += calculate_patch_overlap(window, polygon_coords, threshold)
            skipped += 1 - calculate_patch_overlap(window, polygon_coords, threshold)
    
    return saved, skipped


def process_all_geos():
    """Process all resized geoglif images and extract patches with polygon filtering."""
    assert DATA_DIR.exists(), f"Data directory not found: {DATA_DIR}"
    
    geos_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.endswith('_geos')])
    assert geos_dirs, "No *_geos directories found in resized_geos folder!"
    
    print("=" * 70)
    print("EXTRACTING PATCHES FROM RESIZED GEOGLIF IMAGES")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Found {len(geos_dirs)} area(s) to process\n")
    print(f"Patch settings: {WINDOW_SIZE}x{WINDOW_SIZE}, stride={STRIDE}")
    print(f"Polygon overlap threshold: {THRESHOLD*100:.0f}%\n")
    
    total_images = total_patches = total_originals = total_skipped = 0
    
    for geos_dir in geos_dirs:
        area_name = geos_dir.name.replace('_geos', '')
        tif_files = sorted(geos_dir.glob("*_ortho.tif"))
        
        if not tif_files:
            print(f"[{area_name.upper()}] No .tif files found, skipping...")
            continue
        
        print(f"[{area_name.upper()}] Processing {len(tif_files)} images")
        print("-" * 60)
        
        area_output_dir = OUTPUT_DIR / area_name
        
        for idx, tif_path in enumerate(tif_files, 1):
            base_name = tif_path.stem.replace('_ortho', '')
            geo_output_dir = area_output_dir / base_name
            metadata_path = geos_dir / tif_path.stem.replace('_ortho', '_metadata.json')
            
            print(f"  [{idx}/{len(tif_files)}] {tif_path.name} - loading...", end=" ")
            
            img = load_tif_image(tif_path)
            h, w = img.shape[:2]
            print(f"({h}x{w}) ", end="")
            
            # Load and convert polygon
            polygon_coords = None
            if metadata_path.exists():
                metadata = load_metadata(metadata_path)
                poly_geo = load_polygon_from_metadata(metadata_path)
                if poly_geo and all(k in metadata for k in ['bounds', 'image_shape']):
                    polygon_coords = convert_polygon_geo_to_pixel(poly_geo, metadata['bounds'], metadata['image_shape'])
            
            # Save original image
            geo_output_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img.astype(np.uint8)).save(geo_output_dir / f"{base_name}_original.png")
            
            # Extract and save patches
            patches = extract_patches(img)
            if patches is None:
                print("too small, saved original only... ✓")
                total_originals += 1
            else:
                print("extracting...", end=" ")
                if polygon_coords:
                    saved, skipped = save_patches(patches, polygon_coords, geo_output_dir, THRESHOLD)
                    total_skipped += skipped
                else:
                    # Save all patches without filtering
                    print("(no polygon, saving all) ", end="")
                    rows, cols = patches.shape[0], patches.shape[1]
                    for i in range(rows):
                        for j in range(cols):
                            path = geo_output_dir / f"patch_{i:03d}_{j:03d}.png"
                            Image.fromarray(patches[i, j, 0].astype(np.uint8)).save(path)
                    saved = rows * cols
                    skipped = 0
                
                print(f"✓ ({saved} patches + original" + (f", {skipped} skipped)" if polygon_coords else ")"))
                total_patches += saved
            
            total_images += 1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images: {total_images}")
    print(f"Total patches extracted: {total_patches}")
    print(f"Total patches skipped: {total_skipped}")
    print(f"Total originals: {total_originals}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    process_all_geos()

