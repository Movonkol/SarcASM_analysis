# /env python3
"""
SarcAsM batch (folder) — v5.3
- Zählt n_sarcomeres robust: bevorzugt 'n_vectors', sonst aus 'sarcomere_length_vectors' abgeleitet.
- Modellpfad (model_path) & Rescale-Faktor (rescale_factor) konfigurierbar.
- Falls sarcasm.detect_sarcomeres(rescale_factor=...) nicht unterstützt:
  Fallback: 2D-TIFF wird manuell reskaliert und Pixelgröße angepasst.

Requires:
    pip install sarc-asm
    # Für Fallback-Rescaling (nur falls benötigt):
    pip install scikit-image tifffile

Usage:
    python sarcasm_batch.py --rescale 0.7 --model "C:\pfad\zu\weights.pth"
"""
from pathlib import Path
import os, shutil, csv, argparse, warnings, contextlib
import numpy as np
from sarcasm import Structure

# -------------------- USER SETTINGS --------------------
input_dir = r"C:\Users\Moritz\Downloads\ki67"  # <-- folder with TIFFs
pixelsize_um_per_px = 0.0707008                 # µm/px
out_dir = r"./sarcasm_results"                  # output folder
recurse = False                                 # True -> include subfolders
# Analysis params
threshold_mbands = 0.4
median_filter_radius = 0.25   # µm
linewidth = 0.2               # µm
interp_factor = 4
slen_lims = (1.5, 2.4)        # µm (min, max)

# Konfigurierbar
model_path = None             # z.B. r"C:\...\unetpp_sarcomere.pth"; None = Standard
rescale_factor = 0.7          # 0.5: Downscale, >1.0: Upscale
# -------------------------------------------------------

def try_detect_with_native_args(sarc, model_path, rescale_factor):
    """Probiert detect_sarcomeres mit (model_path, rescale_factor); fällt stufenweise zurück."""
    try:
        if model_path is not None and rescale_factor is not None:
            sarc.detect_sarcomeres(model_path=model_path, rescale_factor=rescale_factor)
            return "native"
    except TypeError:
        pass
    try:
        if model_path is not None:
            sarc.detect_sarcomeres(model_path=model_path)
            return "native_no_rescale"
    except TypeError:
        pass
    try:
        if rescale_factor is not None:
            sarc.detect_sarcomeres(rescale_factor=rescale_factor)
            return "native"
    except TypeError:
        pass
    sarc.detect_sarcomeres()
    return "native_default"

def manual_rescale_2d_tiff(src_path: Path, r: float, out_dir: Path) -> Path:
    """Manuelles 2D-Rescaling als Fallback; gibt Pfad zu Temp-TIFF zurück."""
    if r == 1.0:
        return src_path
    try:
        from tifffile import imread, imwrite
    except Exception as e:
        raise RuntimeError("Bitte 'tifffile' installieren: pip install tifffile") from e
    try:
        from skimage.transform import resize
    except Exception as e:
        raise RuntimeError("Bitte 'scikit-image' installieren: pip install scikit-image") from e
    arr = imread(str(src_path))
    if arr.ndim != 2:
        raise RuntimeError(f"Fallback unterstützt nur 2D-TIFFs: '{src_path.name}' hat {arr.ndim}D.")
    new_h = max(1, int(round(arr.shape[0] * r)))
    new_w = max(1, int(round(arr.shape[1] * r)))
    arr_rs = resize(arr, (new_h, new_w), order=1, preserve_range=True, anti_aliasing=True).astype(arr.dtype)
    tmp_path = out_dir / f"__tmp_rescaled__{src_path.stem}_r{r:.3f}.tif"
    imwrite(str(tmp_path), arr_rs)
    return tmp_path

def write_row_append(csv_file: Path, header, row_dict):
    new_file = not csv_file.exists() or csv_file.stat().st_size == 0
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(row_dict)

def main():
    # -------- CLI --------
    parser = argparse.ArgumentParser(description="Batch SarcAsM: Z-Bänder, Orientierung & Länge aus TIFFs")
    parser.add_argument("--model", dest="cli_model_path", default=None,
                        help="Pfad zu Gewichten (überschreibt model_path im Skript).")
    parser.add_argument("--rescale", dest="cli_rescale", type=float, default=None,
                        help="Rescale-Faktor (z.B. 0.5). Überschreibt rescale_factor.")
    args = parser.parse_args()
    _model_path = args.cli_model_path if args.cli_model_path is not None else model_path
    _rescale_factor = args.cli_rescale if args.cli_rescale is not None else rescale_factor

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

    # Collect files
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files = []
    for pat in patterns:
        files.extend(indir.rglob(pat) if recurse else indir.glob(pat))
    files = sorted({p.resolve() for p in files})
    if not files:
        raise SystemExit(f"No TIFF files found in: {indir}")

    print("[INFO] Settings:")
    print(f"  input_dir         = {indir}")
    print(f"  pixelsize (µm/px) = {pixelsize_um_per_px}")
    print(f"  out_dir           = {out}")
    print(f"  recurse           = {recurse}")
    print(f"  model_path        = {_model_path}")
    print(f"  rescale_factor    = {_rescale_factor}")
    print(f"[INFO] Found {len(files)} image(s). Writing rows to: {csv_path}")

    for i, inp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {inp.name}")
        tmp_created = None
        try:
            # 1) Init & (ggf.) native Rescale
            sarc = Structure(str(inp), pixelsize=pixelsize_um_per_px)
            used_mode = "native_default"
            if _rescale_factor == 1.0:
                used_mode = try_detect_with_native_args(sarc, _model_path, None)
            else:
                try:
                    used_mode = try_detect_with_native_args(sarc, _model_path, _rescale_factor)
                except Exception as e:
                    warnings.warn(f"Native rescale call failed: {e}. Trying manual fallback.")
                    used_mode = "native_failed"
                if used_mode in ("native_no_rescale", "native_default", "native_failed"):
                    try:
                        tmp_created = manual_rescale_2d_tiff(inp, _rescale_factor, out)
                        sarc = Structure(str(tmp_created), pixelsize=pixelsize_um_per_px / _rescale_factor)
                        used_mode = try_detect_with_native_args(sarc, _model_path, None)
                        print(f"    [fallback] manual rescale -> pixelsize={pixelsize_um_per_px / _rescale_factor:.6f} µm/px")
                    except Exception as e:
                        raise RuntimeError(f"Manual rescale fallback failed for {inp.name}: {e}") from e

            # 3) Z-band analysis
            sarc.analyze_z_bands(median_filter_radius=median_filter_radius)
            # 4) Orientation & length
            sarc.analyze_sarcomere_vectors(
                threshold_mbands=threshold_mbands,
                median_filter_radius=median_filter_radius,
                linewidth=linewidth,
                interp_factor=interp_factor,
                slen_lims=slen_lims,
                threshold_sarcomere_mask=0.1,
            )

            # Debug: verfügbare Keys einmal zeigen
            if i == 1:
                print("[DEBUG] data keys:", list(getattr(sarc, "data", {}).keys()))

            # --- Export maps (falls vorhanden) ---
            maps = {
                "orientation_map.tif": getattr(sarc, "file_orientation", None),
                "z_bands_mask.tif": getattr(sarc, "file_zbands", None),
                "m_bands_mask.tif": getattr(sarc, "file_mbands", None),
                "sarcomere_mask.tif": getattr(sarc, "file_sarcomere_mask", None),
            }
            for out_name, src in maps.items():
                if src and os.path.exists(src):
                    shutil.copyfile(src, str(out / f"{inp.stem}_{out_name}"))

            # --- Row for batch CSV ---
            d = getattr(sarc, "data", {}) or {}
            row = {"filename": inp.name, "filepath": str(inp), "detect_mode": used_mode}

            # n_sarcomeres robust bestimmen (an deine Keys angepasst)
            n = d.get("n_sarcomeres")
            if n is None:
                for cand in ("num_sarcomeres", "n_vectors", "n_mbands"):
                    if cand in d:
                        n = d[cand]; break
            if n is None and "sarcomere_length_vectors" in d:
                arr = np.asarray(d["sarcomere_length_vectors"]).astype(float)
                n = int(np.sum(~np.isnan(arr)))
            elif n is None and "pos_vectors" in d:
                n = int(len(np.asarray(d["pos_vectors"])))
            try:
                row["n_sarcomeres"] = int(n) if n is not None and np.isfinite(float(n)) else 0
            except Exception:
                row["n_sarcomeres"] = 0

            # restliche Kennwerte
            for k in keys_of_interest:
                if k == "n_sarcomeres":
                    continue
                v = d.get(k, None)
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

            # konsistenter Header & Append
            header = ["filename", "filepath", "detect_mode"] + keys_of_interest
            write_row_append(csv_path, header, row)

        finally:
            # Temp-Datei aufräumen
            if tmp_created is not None and Path(tmp_created).exists():
                with contextlib.suppress(Exception):
                    Path(tmp_created).unlink()

    print("[OK] Batch done.")

if __name__ == "__main__":
    main()
