# /env python3
"""
SarcAsM batch (folder) — v6.2  (Z-Band-Overlay NACH Filtern)

- Zählt n_sarcomeres robust: bevorzugt 'n_vectors', sonst aus 'sarcomere_length_vectors'.
- Modellpfad (model_path) & Rescale-Faktor (rescale_factor) konfigurierbar.
- Fallback: 2D-TIFF wird manuell reskaliert und Pixelgröße angepasst, falls native Reskalierung nicht geht.
- Overlay: Z-Bänder werden nach den Filtern (Z ∧ finale sarcomere_mask) farbig ins Original gelegt.

Requires:
    pip install sarc-asm
    pip install tifffile scikit-image  # I/O, Resize, Otsu, remove_small_objects

Usage:
    python sarcasm_batch.py --rescale 0.7 --model "C:\\pfad\\zu\\weights.pth"
"""
from pathlib import Path
import os, shutil, csv, argparse, warnings, contextlib
import numpy as np
from sarcasm import Structure

# -------------------- USER SETTINGS --------------------
input_dir = r"C:\Users\Moritz\Downloads\2D_sarcasm\2D_sarcasm" # <-- folder with TIFFs
pixelsize_um_per_px = 0.1417                 # µm/px
out_dir = r"./sarcasm_results"                  # output folder
recurse = False                                 # True -> include subfolders
# Analysis params
threshold_mbands = 0.25
zbands_threshold = 0.3
median_filter_radius = 0.25   # µm
linewidth = 0.2               # µm
interp_factor = 4
slen_lims = (1.5, 2.4)        # µm (min, max)

# Konfigurierbar
model_path = None             # z.B. r"C:\...\unetpp_sarcomere.pth"; None = Standard
rescale_factor = 1.0          # 0.5: Downscale, >1.0: Upscale

# ----------- OVERLAY SETTINGS -----------
make_overlay = True
overlay_color_rgb = (0, 255, 0)     # Farbe der Z-Bänder (R,G,B)
overlay_alpha = 0.55                # 0..1
overlay_suffix = "_overlay_zbands"  # Dateiname-Suffix (.tif)
overlay_mask_threshold = 0.1     # "auto" (Otsu) ODER z.B. 0.3 (fix)
overlay_min_size_px = 0           # Kleinstpartikel entfernen; 0 = aus
# --- FILTERED-Z OVERLAY SETTINGS ---
use_filtered_zmask = True           # Z ∧ finale sarcomere_mask
sarco_mask_threshold = "auto"       # Binarisierung für sarcomere_mask
save_filtered_zmask = True          # gefilterte Z-Maske zusätzlich speichern
# ---------------------------------------
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


# -------------------- OVERLAY HELPERS --------------------
def _to_uint8_grayscale(arr: np.ndarray) -> np.ndarray:
    """Linear nach uint8 [0..255] skaliert (nur für Visualization)."""
    a = np.asarray(arr, dtype=np.float32)
    amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - amin) / (amax - amin)
    return (np.clip(a, 0, 1) * 255.0).astype(np.uint8)


def _ensure_2d(img: np.ndarray) -> np.ndarray:
    """Nimmt 2D oder extrahiert erste Ebene aus (T,Y,X)/(Z,Y,X)."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return img[0]
    raise RuntimeError(f"Overlay: erwartet 2D/3D, bekommen: {img.ndim}D.")


def _resize_mask_to(mask: np.ndarray, target_hw):
    """Nearest-Neighbor-Resize für Masken-Scores auf Ziel-(H,W)."""
    if mask.shape == tuple(target_hw):
        return mask
    try:
        from skimage.transform import resize
    except Exception as e:
        raise RuntimeError("Bitte 'scikit-image' installieren (für Masken-Resize).") from e
    m = resize(mask.astype(float), target_hw, order=0, preserve_range=True, anti_aliasing=False)
    return m


def _binarize_simple(mask2d: np.ndarray, thr="auto", min_size=0) -> np.ndarray:
    """Binarisiert Masken-Scores robust (Otsu oder fixer Schwellwert) und entfernt Kleinstobjekte."""
    m = mask2d.astype(np.float32)
    if thr == "auto":
        try:
            from skimage.filters import threshold_otsu
            t = float(threshold_otsu(m))
            if not np.isfinite(t) or t <= 0:
                t = 0.5 * float(np.nanmax(m))
        except Exception:
            t = 0.5 * float(np.nanmax(m))
    else:
        t = float(thr)
    mb = m > t
    if min_size and min_size > 0:
        try:
            from skimage.morphology import remove_small_objects
            mb = remove_small_objects(mb, min_size=min_size)
        except Exception:
            pass
    return mb


def build_final_zmask(original_path: Path, zmask_path: Path,
                      sarcomere_mask_path,
                      thr_z="auto", thr_sarco="auto", min_size_z=0) -> np.ndarray:
    """Erzeuge finale Z-Maske: (Z) ∧ (finale sarcomere_mask, falls vorhanden)."""
    from tifffile import imread
    base = imread(str(original_path))
    base2d = _ensure_2d(base)
    # Z-Maske laden -> auf Bildgröße -> binarisieren
    z = imread(str(zmask_path))
    z2d = _ensure_2d(z)
    z_rs = _resize_mask_to(z2d, base2d.shape)
    z_bin = _binarize_simple(z_rs, thr=thr_z, min_size=min_size_z)
    # Optional: mit finaler Sarcomere-Maske schneiden
    if sarcomere_mask_path:
        s = imread(str(sarcomere_mask_path))
        s2d = _ensure_2d(s)
        s_rs = _resize_mask_to(s2d, base2d.shape)
        s_bin = _binarize_simple(s_rs, thr=thr_sarco, min_size=0)
        z_bin = np.logical_and(z_bin, s_bin)
    return z_bin


def save_zband_overlay_with_mask(original_path: Path, final_mask_bool: np.ndarray, out_path: Path,
                                 color=(0, 255, 0), alpha=0.55) -> None:
    """Speichert RGB-Overlay ins Originalbild unter Verwendung einer Bool-Maske."""
    from tifffile import imread, imwrite
    base = imread(str(original_path))
    base2d = _ensure_2d(base)
    base8 = _to_uint8_grayscale(base2d)
    rgb = np.stack([base8, base8, base8], axis=-1).astype(np.float32)
    m = final_mask_bool
    if np.any(m):
        color_vec = np.array(color, dtype=np.float32)
        rgb[m] = (1.0 - alpha) * rgb[m] + alpha * color_vec
    rgb_uint8 = rgb.clip(0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from tifffile import imwrite
    imwrite(str(out_path), rgb_uint8, photometric='rgb')


def save_bool_mask(mask_bool: np.ndarray, out_path: Path) -> None:
    """Speichert Bool-Maske als 8-bit TIFF (255 = True)."""
    from tifffile import imwrite
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(out_path), (mask_bool.astype(np.uint8) * 255))
# --------------------------------------------------------


def main():
    # -------- CLI --------
    parser = argparse.ArgumentParser(description="Batch SarcAsM: Z-Bänder, Orientierung & Länge aus TIFFs (mit gefiltertem Z-Overlay)")
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
    print(f"  make_overlay      = {make_overlay} (color={overlay_color_rgb}, alpha={overlay_alpha})")
    print(f"  filtered_z        = {use_filtered_zmask} (thr_z={overlay_mask_threshold}, thr_sarco={sarco_mask_threshold}, min_size={overlay_min_size_px})")
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

            # --- Overlay (Z-Bänder NACH Filtern) ---
            if make_overlay:
                zmask_src = getattr(sarc, "file_zbands", None)
                sarco_mask_src = getattr(sarc, "file_sarcomere_mask", None) if use_filtered_zmask else None
                if zmask_src and os.path.exists(zmask_src):
                    try:
                        final_mask = build_final_zmask(
                            original_path=inp,
                            zmask_path=Path(zmask_src),
                            sarcomere_mask_path=(Path(sarco_mask_src) if (sarco_mask_src and os.path.exists(sarco_mask_src)) else None),
                            thr_z=overlay_mask_threshold,
                            thr_sarco=sarco_mask_threshold,
                            min_size_z=overlay_min_size_px,
                        )
                        # Overlay speichern
                        overlay_path = out / f"{inp.stem}{overlay_suffix}.tif"
                        save_zband_overlay_with_mask(
                            original_path=inp,
                            final_mask_bool=final_mask,
                            out_path=overlay_path,
                            color=overlay_color_rgb,
                            alpha=overlay_alpha,
                        )
                        print(f"    [overlay] saved -> {overlay_path.name}")
                        # Gefilterte Z-Maske optional separat exportieren
                        if save_filtered_zmask:
                            zmask_f_path = out / f"{inp.stem}_z_bands_mask_filtered.tif"
                            save_bool_mask(final_mask, zmask_f_path)
                    except Exception as e:
                        warnings.warn(f"Filtered Z-overlay failed for {inp.name}: {e}")

            # --- Row for batch CSV ---
            d = getattr(sarc, "data", {}) or {}
            row = {"filename": inp.name, "filepath": str(inp), "detect_mode": used_mode}

            # n_sarcomeres robust bestimmen (nach Filtern)
            n = d.get("n_sarcomeres")
            if n is None:
                for cand in ("num_sarcomeres", "n_vectors"):
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
