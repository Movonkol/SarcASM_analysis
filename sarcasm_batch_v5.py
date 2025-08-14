#!/usr/bin/env python3
"""
SarcAsM batch (folder) — v5
- Set variables at top (input_dir, pixelsize, out_dir)
- Processes all .tif/.tiff in the folder (non-recursive by default)
- Saves maps per image; appends one row per image to ONE CSV

Requires:
    pip install sarc-asm
"""
from pathlib import Path
import os, shutil, csv
import numpy as np

# -------------------- USER SETTINGS --------------------
input_dir = r"/mnt/data"           # <-- folder with TIFFs
pixelsize_um_per_px = 0.0901876    # µm/px
out_dir = r"./sarcasm_results"     # output folder
recurse = False                    # True -> include subfolders
# Analysis params
threshold_mbands = 0.5
median_filter_radius = 0.25   # µm
linewidth = 0.2               # µm
interp_factor = 4
slen_lims = (1.5, 2.4)        # µm (min, max)
# -------------------------------------------------------

from sarcasm import Structure

indir = Path(input_dir).expanduser().resolve()
out = Path(out_dir).expanduser().resolve()
out.mkdir(parents=True, exist_ok=True)

# CSV path (single file for all images)
csv_path = out / "sarcasm_batch_basic.csv"
keys_of_interest = [
    "sarcomere_length_mean",
    "sarcomere_length_std",
    "sarcomere_orientation_mean",
    "sarcomere_orientation_std",
    "sarcomere_oop",
    "n_sarcomeres",
]

def write_row_append(csv_file, header, row_dict):
    new_file = not csv_file.exists() or csv_file.stat().st_size == 0
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(row_dict)

# Collect files
patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
files = []
for pat in patterns:
    files.extend(indir.rglob(pat) if recurse else indir.glob(pat))
files = sorted({p.resolve() for p in files})

if not files:
    raise SystemExit(f"No TIFF files found in: {indir}")

print(f"[INFO] Found {len(files)} image(s). Writing rows to: {csv_path}")

for i, inp in enumerate(files, 1):
    print(f"[{i}/{len(files)}] {inp.name}")
    # 1) Init
    sarc = Structure(str(inp), pixelsize=pixelsize_um_per_px)
    # 2) Detect sarcomeres
    sarc.detect_sarcomeres()
    # 3) Z-band analysis
    sarc.analyze_z_bands(median_filter_radius=median_filter_radius)
    # 4) Orientation & length
    sarc.analyze_sarcomere_vectors(
        threshold_mbands=threshold_mbands,
        median_filter_radius=median_filter_radius,
        linewidth=linewidth,
        interp_factor=interp_factor,
        slen_lims=slen_lims,
        threshold_sarcomere_mask=0.1
    )

    # --- Export maps ---
    maps = {
        "orientation_map.tif": getattr(sarc, "file_orientation", None),
        "z_bands_mask.tif": getattr(sarc, "file_zbands", None),
        "m_bands_mask.tif": getattr(sarc, "file_mbands", None),
        "sarcomere_mask.tif": getattr(sarc, "file_sarcomere_mask", None),
    }
    for out_name, src in maps.items():
        if src and os.path.exists(src):
            dst = out / f"{inp.stem}_{out_name}"
            shutil.copyfile(src, str(dst))

    # --- Row for batch CSV ---
    row = {"filename": inp.name, "filepath": str(inp)}
    for k in keys_of_interest:
        v = sarc.data.get(k, None)
        if v is None:
            continue
        try:
            if np.ndim(v) == 0:
                row[k] = float(v)
            else:
                arr = np.asarray(v).astype(float)
                row[k] = float(np.nanmean(arr))
        except Exception:
            row[k] = str(v)

    # Ensure consistent header: filename + metrics
    header = ["filename", "filepath"] + keys_of_interest
    write_row_append(csv_path, header, row)

print("[OK] Batch done.")