"""
Multiprocessing Example für SarcAsM Batch
Zeigt, wie die Batch-Verarbeitung mit Multiprocessing optimiert werden kann

⚠️ WICHTIG: Fiji/PyImageJ ist NICHT thread-safe!
Lösung: enable_fiji_via_pyimagej = False bei Multiprocessing

Erwarteter Speedup: 4-8x (abhängig von CPU-Kernen)
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Dict
import logging
import warnings

# ============================================================
# WORKER-INITIALISIERUNG (pro Prozess)
# ============================================================

def init_worker():
    """
    Wird für jeden Worker-Prozess einmal aufgerufen

    Wichtig für:
    - Fiji-Deaktivierung (nicht thread-safe!)
    - Logger-Setup pro Prozess
    - Seed für Zufallszahlen
    """
    # Fiji/PyImageJ DEAKTIVIEREN (nicht prozess-sicher!)
    import os
    os.environ["SARCASM_DISABLE_FIJI"] = "1"

    # Logging für Worker konfigurieren (verhindert Durcheinander)
    # Option 1: Nur FileHandler, kein StreamHandler
    # Option 2: Process-spezifisches Log
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker-{mp.current_process().name}] %(message)s'
    )

    # Optional: Seed für Reproduzierbarkeit
    import numpy as np
    np.random.seed(None)  # Jeder Worker eigener Seed


# ============================================================
# WRAPPER FÜR PROCESS_ONE_IMAGE
# ============================================================

def process_one_image_wrapper(args: Tuple[Path, Path]) -> Tuple[Path, dict]:
    """
    Wrapper für process_one_image() - kompatibel mit Multiprocessing

    Args:
        args: Tuple von (input_path, output_dir)

    Returns:
        Tuple von (input_path, result_dict)

    ⚠️ Wichtig: Muss auf Top-Level sein (nicht in Klasse/Funktion verschachtelt)
    """
    inp, out = args

    try:
        # Hier wird die Original-Funktion aufgerufen
        # (muss importiert oder hier definiert sein)
        from sarcasm_batch_v5 import process_one_image

        row = process_one_image(inp, out)
        return (inp, row)

    except BaseException as e:
        warnings.warn(f"[{inp.name}] Worker failed: {e}")
        return (inp, {})


# ============================================================
# MAIN-FUNKTION MIT MULTIPROCESSING
# ============================================================

def main_multiprocessing(
    input_dir: str,
    output_dir: str,
    n_workers: int = None,
    recurse: bool = False
):
    """
    Hauptfunktion mit Multiprocessing

    Args:
        input_dir: Input-Verzeichnis mit TIFFs
        output_dir: Output-Verzeichnis für Ergebnisse
        n_workers: Anzahl Worker (None = automatisch CPU_count - 1)
        recurse: Rekursiv in Unterordnern suchen
    """
    LOG = logging.getLogger("sarcasm_batch")

    # Setup
    indir = Path(input_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # CSV Header
    header = [
        "filename","filepath","detect_mode","preproc",
        "pixelsize_um_per_px","pixelsize_src",
        "n_sarcomeres","n_sarcomeres_overlay",
        "sarcomere_length_mean","sarcomere_length_std",
        "slen_median","slen_p10","slen_p90","slen_n_valid",
        "orientation_mean","orientation_std","sarcomere_oop",
        "orient_median","orient_p10","orient_p90","orient_n_valid",
    ]
    csv_path = out / "sarcasm_batch_basic.csv"

    # Dateien sammeln (optimierte Version)
    from optimized_helpers import collect_tiff_files_optimized
    files = collect_tiff_files_optimized(indir, recurse=recurse)

    if not files:
        LOG.error(f"No TIFF files found in: {indir}")
        return

    # Worker-Count bestimmen (n-1 Kerne für System-Stabilität)
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    LOG.info(f"[INFO] Processing {len(files)} images with {n_workers} workers")
    LOG.warning("⚠️  Fiji/PyImageJ wird deaktiviert (nicht multiprocessing-safe)")

    # Multiprocessing mit Progress-Tracking
    completed = 0
    results = []

    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as executor:
        # Alle Jobs starten
        future_to_file = {
            executor.submit(process_one_image_wrapper, (f, out)): f
            for f in files
        }

        # Ergebnisse sammeln (as_completed zeigt Progress)
        for future in as_completed(future_to_file):
            completed += 1
            filepath = future_to_file[future]

            try:
                inp, row = future.result(timeout=300)  # 5min Timeout pro Bild
                LOG.info(f"[{completed}/{len(files)}] ✓ {filepath.name}")

                if row:
                    # Zur CSV schreiben (thread-safe)
                    from sarcasm_batch_v5 import write_row_append
                    write_row_append(csv_path, header, row)
                    results.append(row)

            except TimeoutError:
                LOG.error(f"[{completed}/{len(files)}] ✗ {filepath.name} - TIMEOUT")

            except Exception as e:
                LOG.error(f"[{completed}/{len(files)}] ✗ {filepath.name} - {e}")

    LOG.info(f"[OK] Batch done. Processed {len(results)}/{len(files)} successfully.")


# ============================================================
# ALTERNATIVE: CHUNKWISE PROCESSING
# ============================================================

def main_multiprocessing_chunked(
    input_dir: str,
    output_dir: str,
    n_workers: int = None,
    chunk_size: int = 10,
    recurse: bool = False
):
    """
    Alternative mit Chunked Processing

    Vorteile:
    - Bessere Memory-Kontrolle (nicht alle Jobs gleichzeitig)
    - Frühere Fehler-Erkennung
    - Besseres Progress-Reporting

    Args:
        chunk_size: Anzahl Bilder pro Batch
    """
    from optimized_helpers import collect_tiff_files_optimized

    LOG = logging.getLogger("sarcasm_batch")

    # Setup
    indir = Path(input_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    files = collect_tiff_files_optimized(indir, recurse=recurse)
    if not files:
        LOG.error(f"No TIFF files found in: {indir}")
        return

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    LOG.info(f"[INFO] Processing {len(files)} images in chunks of {chunk_size}")

    # Verarbeitung in Chunks
    total_processed = 0
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i+chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (len(files) + chunk_size - 1) // chunk_size

        LOG.info(f"=== Chunk {chunk_num}/{total_chunks} ({len(chunk)} files) ===")

        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as executor:
            futures = {
                executor.submit(process_one_image_wrapper, (f, out)): f
                for f in chunk
            }

            for future in as_completed(futures):
                total_processed += 1
                filepath = futures[future]

                try:
                    inp, row = future.result(timeout=300)
                    LOG.info(f"  [{total_processed}/{len(files)}] ✓ {filepath.name}")

                    if row:
                        from sarcasm_batch_v5 import write_row_append
                        csv_path = out / "sarcasm_batch_basic.csv"
                        header = ["filename","filepath","detect_mode","preproc",
                                 "pixelsize_um_per_px","pixelsize_src",
                                 "n_sarcomeres","n_sarcomeres_overlay"]
                        write_row_append(csv_path, header, row)

                except Exception as e:
                    LOG.error(f"  [{total_processed}/{len(files)}] ✗ {filepath.name} - {e}")

    LOG.info(f"[OK] All chunks processed. Total: {total_processed}/{len(files)}")


# ============================================================
# BENCHMARK: Single-Thread vs Multi-Thread
# ============================================================

def benchmark_speedup(input_dir: str, output_dir: str, sample_size: int = 10):
    """
    Benchmark: Vergleicht Single-Thread vs Multiprocessing

    Args:
        input_dir: Test-Verzeichnis
        output_dir: Output
        sample_size: Anzahl Bilder für Test
    """
    import time
    from optimized_helpers import collect_tiff_files_optimized

    print("=== Speedup Benchmark ===\n")

    # Dateien sammeln
    files = collect_tiff_files_optimized(Path(input_dir), recurse=False)
    files = files[:sample_size]  # Nur Sample

    if len(files) < 2:
        print("⚠️  Need at least 2 files for benchmark")
        return

    print(f"Testing with {len(files)} files...\n")

    # Test 1: Single-Thread (Original)
    print("1. Single-Thread Processing:")
    start = time.perf_counter()

    # Original-Loop simulieren
    for f in files:
        try:
            from sarcasm_batch_v5 import process_one_image
            _ = process_one_image(f, Path(output_dir))
        except Exception as e:
            print(f"  Error: {e}")

    single_time = time.perf_counter() - start
    print(f"   Time: {single_time:.2f}s\n")

    # Test 2: Multiprocessing
    print("2. Multiprocessing:")
    n_workers = max(1, mp.cpu_count() - 1)
    print(f"   Workers: {n_workers}")

    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as executor:
        futures = [
            executor.submit(process_one_image_wrapper, (f, Path(output_dir)))
            for f in files
        ]
        for future in as_completed(futures):
            try:
                _ = future.result(timeout=300)
            except Exception:
                pass

    multi_time = time.perf_counter() - start
    print(f"   Time: {multi_time:.2f}s\n")

    # Ergebnis
    speedup = single_time / multi_time if multi_time > 0 else 0
    print(f"=== Results ===")
    print(f"Single-Thread: {single_time:.2f}s")
    print(f"Multi-Thread:  {multi_time:.2f}s")
    print(f"Speedup:       {speedup:.2f}x")
    print(f"Efficiency:    {speedup/n_workers*100:.1f}% (ideal: 100%)")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SarcAsM Batch with Multiprocessing"
    )
    parser.add_argument("input_dir", help="Input directory with TIFF files")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("-w", "--workers", type=int, default=None,
                       help="Number of workers (default: CPU_count - 1)")
    parser.add_argument("-r", "--recurse", action="store_true",
                       help="Recurse into subdirectories")
    parser.add_argument("--chunked", action="store_true",
                       help="Use chunked processing")
    parser.add_argument("--chunk-size", type=int, default=10,
                       help="Chunk size for chunked processing")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run speedup benchmark")

    args = parser.parse_args()

    if args.benchmark:
        benchmark_speedup(args.input_dir, args.output_dir)
    elif args.chunked:
        main_multiprocessing_chunked(
            args.input_dir, args.output_dir,
            n_workers=args.workers,
            chunk_size=args.chunk_size,
            recurse=args.recurse
        )
    else:
        main_multiprocessing(
            args.input_dir, args.output_dir,
            n_workers=args.workers,
            recurse=args.recurse
        )


# ============================================================
# USAGE EXAMPLES
# ============================================================

"""
USAGE:

1. Standard Multiprocessing:
    python multiprocessing_example.py ./input ./output

2. Mit 4 Workern:
    python multiprocessing_example.py ./input ./output -w 4

3. Rekursiv:
    python multiprocessing_example.py ./input ./output -r

4. Chunked Processing:
    python multiprocessing_example.py ./input ./output --chunked --chunk-size 20

5. Benchmark:
    python multiprocessing_example.py ./input ./output --benchmark


INTEGRATION in sarcasm_batch_v5.py:

    # In main():
    USE_MULTIPROCESSING = True  # Toggle
    N_WORKERS = None  # Auto-detect

    if USE_MULTIPROCESSING:
        from multiprocessing_example import main_multiprocessing
        main_multiprocessing(input_dir, out_dir, n_workers=N_WORKERS, recurse=recurse)
    else:
        # Original single-thread loop
        for f in files:
            ...

"""
