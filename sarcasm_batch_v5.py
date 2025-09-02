#!/usr/bin/env python3
"""
SarcAsM batch (folder) — v5.1
- NEU: Modellpfad (model_path) & Rescale-Faktor (rescale_factor) konfigurierbar.
- Falls sarcasm.detect_sarcomeres(rescale_factor=...) nicht unterstützt:
  Fallback: Bild wird (2D) manuell reskaliert und mit angepasster Pixelgröße analysiert.

Requires:
    pip install sarc-asm
    # Für Fallback-Rescaling (nur falls benötigt):
    pip install scikit-image tifffile
"""
from pathlib import Path
import os, shutil, csv, argparse, tempfile, warnings
import numpy as np

# -------------------- USER SETTINGS --------------------
input_dir = r"C:\Users\Moritz\Downloads\ki67" # <-- folder with TIFFs
pixelsize_um_per_px = 0.0707008   # µm/px
out_dir = r"./sarcasm_results"     # output folder
recurse = False                    # True -> include subfolders
# Analysis params
threshold_mbands = 0.4
median_filter_radius = 0.25   # µm
linewidth = 0.2               # µm
interp_factor = 4
slen_lims = (1.5, 2.4)        # µm (min, max)

# >>> NEU: Modell & Rescale konfigurierbar
model_path = None             # z.B. r"C:\pfad\zu\weights\unetpp_sarcomere.pth"; None = Standardmodell
rescale_factor = 0.7        # z.B. 0.5 für Downscale, >1.0 für Upscale
# -------------------------------------------------------

from sarcasm import Structure


def try_detect_with_native_args(sarc, model_path, rescale_factor):
    """
    Versucht detect_sarcomeres mit (model_path, rescale_factor).
    Fällt stufenweise zurück, wenn Signatur die Args nicht unterstützt.
    Rückgabe: ("native" | "native_no_rescale" | "native_default")
    """
    # Voller Versuch
    try:
        if model_path is not None and rescale_factor is not None:
            sarc.detect_sarcomeres(model_path=model_path, rescale_factor=rescale_factor)
            return "native"
    except TypeError:
        pass

    # Nur model_path
    try:
        if model_path is not None:
            sarc.detect_sarcomeres(model_path=model_path)
            return "native_no_rescale"
    except TypeError:
        pass

    # Nur rescale_factor
    try:
        if rescale_factor is not None:
            sarc.detect_sarcomeres(rescale_factor=rescale_factor)
            return "native"
    except TypeError:
        pass

    # Ganz ohne Zusatz-Args
    sarc.detect_sarcomeres()
    return "native_default"


def manual_rescale_2d_tiff(src_path: Path, r: float, out_dir: Path) -> Path:
    """
    Skaliert ein 2D-TIFF manuell mit skimage (Fallback) und schreibt temporär.
    Gibt Pfad zur temporären TIFF zurück.
    """
    if r == 1.0:
        return src_path

    try:
        from tifffile import imread, imwrite
    except Exception as e:
        raise RuntimeError(
            "Für den manuellen Rescale-Fallback wird 'tifffile' benötigt. "
            "Installiere es mit: pip install tifffile"
        ) from e
    try:
        from skimage.transform import resize
    except Exception as e:
        raise RuntimeError(
            "Für den manuellen Rescale-Fallback wird 'scikit-image' benötigt. "
            "Installiere es mit: pip install scikit-image"
        ) from e

    arr = imread(str(src_path))
    if arr.ndim != 2:
        raise RuntimeError(
            f"Manueller Rescale-Fallback unterstützt nur 2D-TIFFs. "
            f"Bild '{src_path.name}' hat {arr.ndim} Dimension(en)."
        )

    new_h = max(1, int(round(arr.shape[0] * r)))
    new_w = max(1, int(round(arr.shape[1] * r)))
    arr_rs = resize(
        arr, (new_h, new_w), order=1, preserve_range=True, anti_aliasing=True
    ).astype(arr.dtype)

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
    parser = argparse.ArgumentParser(
        description="Batch SarcAsM: Z-Bänder, Orientierung & Länge aus TIFFs"
    )
    parser.add_argument("--model", dest="cli_model_path", default=None,
                        help="Pfad zu Gewichten (überschreibt model_path aus dem Skript).")
    parser.add_argument("--rescale", dest="cli_rescale", type=float, default=None,
                        help="Rescale-Faktor (z.B. 0.5). Überschreibt rescale_factor aus dem Skript.")
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
            # 1) Init & ggf. native Rescale
            sarc = Structure(str(inp), pixelsize=pixelsize_um_per_px)

            used_mode = "native_default"
            if _rescale_factor == 1.0:
                used_mode = try_detect_with_native_args(
                    sarc, _model_path, None
                )
            else:
                # Erst versuchen, ob detect_sarcomeres ein rescale_factor-Arg versteht
                try:
                    used_mode = try_detect_with_native_args(
                        sarc, _model_path, _rescale_factor
                    )
                except Exception as e:
                    # Wenn das (aus anderen Gründen) schiefgeht, machen wir Fallback
                    warnings.warn(f"Native rescale call failed: {e}. Trying manual fallback.")
                    used_mode = "native_failed"

                # Falls der native Weg kein rescale unterstützt -> manueller Fallback
                if used_mode in ("native_no_rescale", "native_default", "native_failed"):
                    # Manuell reskalieren (2D), Pixelgröße anpassen
                    try:
                        tmp_created = manual_rescale_2d_tiff(inp, _rescale_factor, out)
                        sarc = Structure(str(tmp_created), pixelsize=pixelsize_um_per_px / _rescale_factor)
                        # Jetzt ohne rescale-Arg detektieren (mit/ohne model_path)
                        used_mode = try_detect_with_native_args(
                            sarc, _model_path, None
                        )
                        print(f"    [fallback] manual rescale -> pixelsize={pixelsize_um_per_px / _rescale_factor:.6f} µm/px")
                    except Exception as e:
                        raise RuntimeError(
                            f"Manual rescale fallback failed for {inp.name}: {e}"
                        ) from e

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
            row = {"filename": inp.name, "filepath": str(inp), "detect_mode": used_mode}
            for k in keys_of_interest:
                v = getattr(sarc, "data", {}).get(k, None)
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

            # Ensure consistent header
            header = ["filename", "filepath", "detect_mode"] + keys_of_interest
            write_row_append(csv_path, header, row)

        finally:
            # Temp-Datei aufräumen
            if tmp_created is not None and Path(tmp_created).exists():
                with contextlib.suppress(Exception):
                    Path(tmp_created).unlink()

    print("[OK] Batch done.")


if __name__ == "__main__":
    import contextlib
    main()
