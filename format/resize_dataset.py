"""Resize geoglif images using LCI method with georeferencing preserved."""

from pathlib import Path
import json
import argparse
import numpy as np
from PIL import Image
import rasterio
from scipy.fft import idct
from rasterio.transform import Affine


def lci(I_in, *args):
    """Lagrange-Chebyshev Interpolation image resizing."""
    I = np.asarray(I_in)
    if I.ndim == 2:
        n1, n2 = I.shape
        c = 1
    elif I.ndim == 3 and I.shape[2] in (3, 4):
        n1, n2 = I.shape[:2]
        c = 3
        I = I[:, :, :3]
    else:
        raise ValueError("I_in must be (H,W) or (H,W,3) (optionally (H,W,4) RGBA).")

    if len(args) == 2:
        mi, ni = args
    elif len(args) == 1:
        scale = args[0]
        mi = int(round(scale * n1))
        ni = int(round(scale * n2))
    else:
        raise ValueError("Usage: lci(I_in, mi, ni) or lci(I_in, scale)")

    assert mi > 0 and ni > 0, "Target size must be positive."

    I = I.astype(np.float64, copy=False)
    eta = (2 * np.arange(1, mi + 1) - 1) * np.pi / (2 * mi)
    csi = (2 * np.arange(1, ni + 1) - 1) * np.pi / (2 * ni)

    k1 = np.arange(0, n1)[:, None]
    T1 = np.cos(k1 * eta[None, :]) * np.sqrt(2.0 / n1)
    T1[0, :] = np.sqrt(1.0 / n1)

    k2 = np.arange(0, n2)[:, None]
    T2 = np.cos(k2 * csi[None, :]) * np.sqrt(2.0 / n2)
    T2[0, :] = np.sqrt(1.0 / n2)

    lx = idct(T1, type=2, norm="ortho", axis=0)
    ly = idct(T2, type=2, norm="ortho", axis=0)

    to_uint8 = lambda x: np.clip(np.rint(x), 0, 255).astype(np.uint8)

    if c == 3:
        I_fin = np.empty((mi, ni, 3), dtype=np.uint8)
        for ch in range(3):
            I_fin[:, :, ch] = to_uint8((lx.T @ I[:, :, ch]) @ ly)
    else:
        I_fin = to_uint8((lx.T @ I) @ ly)

    return I_fin


def save_georeferenced_tif(image_array, output_path, source_tif_path, scale_factor=None, target_size=None):
    """Save resized image as georeferenced GeoTIFF."""
    output_path = Path(output_path)
    
    with rasterio.open(source_tif_path) as src:
        src_crs, src_transform = src.crs, src.transform
        src_height, src_width = src.height, src.width
    
    if scale_factor is not None:
        scale_x = scale_y = scale_factor
    elif target_size is not None:
        target_height, target_width = target_size
        scale_y = target_height / src_height
        scale_x = target_width / src_width
    else:
        raise ValueError("Either scale_factor or target_size must be provided")
    
    new_transform = Affine(
        src_transform.a / scale_x, src_transform.b, src_transform.c,
        src_transform.d, src_transform.e / scale_y, src_transform.f
    )
    
    count = 1 if image_array.ndim == 2 else image_array.shape[2]
    data = image_array if image_array.ndim == 2 else np.transpose(image_array, (2, 0, 1))
    
    with rasterio.open(output_path, 'w', driver='GTiff', height=image_array.shape[0],
                       width=image_array.shape[1], count=count, dtype=image_array.dtype,
                       crs=src_crs, transform=new_transform, compress='lzw') as dst:
        if count == 1:
            dst.write(data, 1)
        else:
            for i in range(count):
                dst.write(data[i], i + 1)


def lci_georeferenced(source_tif_path, output_path, scale_factor):
    """Apply LCI resizing and save with georeferencing."""
    with rasterio.open(source_tif_path) as src:
        img = np.stack([src.read(i) for i in [1, 2, 3]], axis=-1) if src.count >= 3 else src.read(1)
    
    img_resized = lci(img, scale_factor)
    save_georeferenced_tif(img_resized, output_path, source_tif_path, scale_factor=scale_factor)
    return img_resized


AREA_SCALES = {'unita': 0.886, 'lluta': 0.218, 'chugchug': 0.18}
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "resized_geos"

DATA_DIR = DEFAULT_DATA_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def update_metadata_json(metadata_path, output_path, scale_factor):
    """Update metadata with new image dimensions."""
    metadata = json.loads(metadata_path.read_text())
    
    if 'image_shape' in metadata:
        metadata['image_shape']['width'] = int(round(metadata['image_shape']['width'] * scale_factor))
        metadata['image_shape']['height'] = int(round(metadata['image_shape']['height'] * scale_factor))
    
    if 'file_size_mb' in metadata:
        metadata['file_size_mb'] *= scale_factor ** 2
    
    metadata['resized'] = True
    metadata['scale_factor'] = scale_factor
    
    output_path.write_text(json.dumps(metadata, indent=2))


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Resize geoglif images using LCI method.')
    parser.add_argument('--data-dir', type=Path, default=None, help=f'Data directory (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--output-dir', type=Path, default=None, help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    return parser.parse_args()


def resize_all_geos():
    """Process all geoglif images and resize with georeferencing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    geos_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.endswith('_geos2')])
    assert geos_dirs, "No *_geos2 directories found in data folder!"
    
    print("=" * 70)
    print("RESIZING ALL GEOGLIF IMAGES WITH LCI + GEOREFERENCING")
    print("=" * 70)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Found {len(geos_dirs)} area(s) to process")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    total_processed = total_skipped = 0
    
    for geos_dir in geos_dirs:
        area_name = geos_dir.name.replace('_geos2', '')
        scale_factor = AREA_SCALES.get(area_name)
        
        if not scale_factor:
            print(f"\n[{area_name.upper()}] No scale factor defined, skipping...")
            continue
        
        tif_files = sorted(geos_dir.glob("*_ortho.tif"))
        if not tif_files:
            print(f"\n[{area_name.upper()}] No .tif files found, skipping...")
            continue
        
        print(f"\n[{area_name.upper()}] Processing {len(tif_files)} images (scale: {scale_factor})")
        print("-" * 60)
        
        area_output_dir = OUTPUT_DIR / f"{area_name}_geos"
        area_output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, tif_path in enumerate(tif_files, 1):
            output_path = area_output_dir / tif_path.name
            
            if output_path.exists():
                print(f"  [{idx}/{len(tif_files)}] {tif_path.name} - already exists, skipping")
                total_skipped += 1
                continue
            
            print(f"  [{idx}/{len(tif_files)}] {tif_path.name} - resizing...", end=" ")
            
            metadata_path = geos_dir / tif_path.stem.replace('_ortho', '_metadata.json')
            output_metadata_path = area_output_dir / metadata_path.name
            
            img_resized = lci_georeferenced(str(tif_path), str(output_path), scale_factor)
            Image.fromarray(img_resized).save(area_output_dir / tif_path.stem.replace('_ortho', '_resized.png'))
            
            print("✓", end=" ")
            
            if metadata_path.exists():
                print("+ metadata...", end=" ")
                update_metadata_json(metadata_path, output_metadata_path, scale_factor)
                print("✓")
            else:
                print("(no metadata)")
            
            total_processed += 1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images processed: {total_processed}")
    print(f"Total images skipped: {total_skipped}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.data_dir:
        DATA_DIR = args.data_dir.resolve()
        print(f"Using custom data directory: {DATA_DIR}")
    
    if args.output_dir:
        OUTPUT_DIR = args.output_dir.resolve()
        print(f"Using custom output directory: {OUTPUT_DIR}")
    
    resize_all_geos()