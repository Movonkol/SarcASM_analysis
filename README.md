# SarcAsM Batch (Folder) — v5

**Batch-processing pipeline for sarcomere analysis**  
This script uses the [`sarc-asm`](https://pypi.org/project/sarc-asm/) toolkit to automatically detect and analyze sarcomeres in single-channel TIFF microscopy images.  
It processes all images in a given folder, generates orientation/Z-band/M-band/mask maps, and compiles key quantitative metrics into a single CSV file.

---

## Input image requirements

- **TIFF format** (`.tif` / `.tiff`) — 8-bit or 16-bit grayscale.
- **Single channel only**: The script expects an image containing **only the sarcomere signal** (no RGB, no extra channels).  
  If your microscopy export contains multiple channels, split them first in ImageJ/Fiji or your acquisition software and save only the relevant sarcomere channel as a separate `.tif`.
- Avoid compression; use uncompressed TIFFs for compatibility.
- Find the Pixel Size of your Image in ImageJ/Fiji
  Image → Properties (Shift+P) → Width, Height, Pixel size, Unit

---

## Quick start

1) **Install**
```bash
pip install sarc-asm numpy
```

2) **Save the script** (e.g. as `sarcasm_batch_v5.py`) and edit the **USER SETTINGS** block at the top:
```python
input_dir = r"C:\Users\Moritz\Downloads\MM.4"  # folder with TIFFs
pixelsize_um_per_px = 0.0901876                # µm per pixel
out_dir = r"./sarcasm_results"                 # output folder
recurse = False                                # True -> include subfolders
# Analysis params
threshold_mbands = 0.5
median_filter_radius = 0.25   # µm
linewidth = 0.2               # µm
interp_factor = 4
slen_lims = (1.5, 2.4)        # µm (min, max)
```

3) **Run**
```bash
python sarcasm_batch_v5.py
```

---

## What the script does

For each `.tif/.tiff` in `input_dir` (non-recursive unless `recurse=True`):

1. Initialize a `sarc-asm` `Structure` with your **pixel size** (`µm/px`).
2. `detect_sarcomeres()`  
3. `analyze_z_bands(median_filter_radius=...)`  
4. `analyze_sarcomere_vectors(...)` using your thresholds/limits.
5. **Copy outputs** for that image into `out_dir`:
   - `{stem}_orientation_map.tif`
   - `{stem}_z_bands_mask.tif`
   - `{stem}_m_bands_mask.tif`
   - `{stem}_sarcomere_mask.tif`
6. **Append one row** to `sarcasm_batch_basic.csv` with:
   - `filename`, `filepath`
   - `sarcomere_length_mean`, `sarcomere_length_std`
   - `sarcomere_orientation_mean`, `sarcomere_orientation_std`
   - `sarcomere_oop`
   - `n_sarcomeres`

If a metric is an array, the script stores its **NaN-mean**; scalars are cast to `float`.

---

## Parameters (at the top of the script)

| Name | Type | Unit | Purpose / Notes |
|---|---|---|---|
| `input_dir` | path | — | Folder containing `.tif/.tiff` images. |
| `pixelsize_um_per_px` | float | µm/px | **Critical.** Your microscope’s pixel size. |
| `out_dir` | path | — | Where maps and CSV are written. Auto-created. |
| `recurse` | bool | — | `True` to include subfolders (`rglob`). |
| `threshold_mbands` | float | — | Threshold for M-band detection in vector analysis. |
| `median_filter_radius` | float | µm | Median filter radius used in Z-band & vector steps. |
| `linewidth` | float | µm | Line width used for vector analysis. |
| `interp_factor` | int | — | Interpolation factor for vector analysis. |
| `slen_lims` | (min,max) | µm | Accepted sarcomere length range. |

---

## Output

```
out_dir/
├─ sarcasm_batch_basic.csv
├─ image1_orientation_map.tif
├─ image1_z_bands_mask.tif
├─ image1_m_bands_mask.tif
├─ image1_sarcomere_mask.tif
└─ ... (repeated per image)
```

- **CSV** contains one row per processed image with consistent headers.
- **Maps** are copied from the `sarc-asm` temp outputs and renamed with the input stem.

---

## Tips & troubleshooting

- **No TIFF files found**: Check `input_dir`, file extensions, and `recurse`.
- **Incorrect metrics**: Verify `pixelsize_um_per_px`; wrong pixel size skews lengths.
- **Too few/too many sarcomeres**: Adjust `threshold_mbands`, `median_filter_radius`, `linewidth`, and `slen_lims`.
- **Performance**: Set `recurse=False` to limit file count; reduce `interp_factor` if needed.

---

## Requirements

- Python 3.8+
- Packages: `sarc-asm`, `numpy` (installed via `pip install sarc-asm numpy`)
- OS: Windows/macOS/Linux (paths shown are examples)


