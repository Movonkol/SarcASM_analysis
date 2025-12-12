# SarcAsM Batch - Konkrete Optimierungsvorschläge

## 🚀 Priorität 1: Multiprocessing (4-8x Speedup)

### Aktueller Code (Single-Thread):
```python
# Zeile 719-727
for i, f in enumerate(files, 1):
    LOG.info(f"[{i}/{len(files)}] {f.name}")
    try:
        row = process_one_image(f, out)
        if row: write_row_append(csv_path, header, row)
    except BaseException as e:
        warnings.warn(f"[{f.name}] failed: {e}")
        continue
```

### Optimierter Code mit Multiprocessing:

```python
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_one_image_wrapper(inp: Path, out: Path) -> Tuple[Path, dict]:
    """Wrapper für Multiprocessing - gibt (filepath, result) zurück"""
    try:
        row = process_one_image(inp, out)
        return (inp, row)
    except BaseException as e:
        warnings.warn(f"[{inp.name}] failed: {e}")
        return (inp, {})

def main_optimized():
    # ... (Setup wie bisher)

    # CPU-Kerne ermitteln (minus 1 für System-Stabilität)
    n_workers = max(1, mp.cpu_count() - 1)
    LOG.info(f"[INFO] Processing {len(files)} images with {n_workers} workers")

    # Multiprocessing mit Progress-Tracking
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Jobs starten
        futures = {
            executor.submit(process_one_image_wrapper, f, out): f
            for f in files
        }

        # Ergebnisse sammeln (mit Progress)
        for future in as_completed(futures):
            completed += 1
            filepath = futures[future]
            LOG.info(f"[{completed}/{len(files)}] {filepath.name}")

            try:
                inp, row = future.result()
                if row:
                    write_row_append(csv_path, header, row)
            except Exception as e:
                LOG.error(f"[{filepath.name}] Processing failed: {e}")
                continue

    LOG.info("[OK] Batch done.")
```

### ⚠️ Wichtige Hinweise für Multiprocessing:

1. **Fiji/PyImageJ ist NICHT thread-safe!** Lösung:
```python
# In process_one_image() ODER in worker-init:
def init_worker():
    """Jeder Worker initialisiert eigenes Fiji"""
    global _IJ_CTX
    _IJ_CTX = None  # Erzwingt Neuinitialisierung pro Prozess

# Dann:
with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as executor:
    ...
```

2. **Logging muss angepasst werden:**
```python
def init_worker_logging():
    """Setup logging für jeden Worker"""
    # Logger per Prozess neu konfigurieren
    # Oder: nur FileHandler, kein StreamHandler (vermeidet Durcheinander)
    pass
```

---

## 📁 Priorität 2: TIFF nur einmal laden (20-40% Speedup)

### Problem: TIFF wird 3x gelesen!
```python
# 1. In detect_pixelsize_um_per_px() - Zeile 269
with tfi.TiffFile(str(tiff_path)) as tf:
    # Metadaten lesen...

# 2. In Structure() - Zeile 542
sarc = Structure(str(cand_img), pixelsize=px_um)  # Lädt TIFF intern!

# 3. In build_final_zmask_exact() - Zeile 474
base = imread(str(original_path))  # Nochmal!
```

### Optimierte Lösung: Metadaten-Cache

```python
from dataclasses import dataclass
from typing import Optional, Tuple
import tifffile as tfi

@dataclass
class TiffMetadata:
    """Gecachte TIFF-Metadaten"""
    path: Path
    width: int
    height: int
    shape: Tuple[int, ...]
    pixelsize_um: Optional[float]
    pixelsize_src: str
    # Optional: kleine Metadaten-Dicts
    imagej_metadata: Optional[dict] = None
    ome_metadata: Optional[str] = None

def extract_tiff_metadata(tiff_path: Path) -> TiffMetadata:
    """Liest TIFF einmal und extrahiert alle benötigten Metadaten"""
    with tfi.TiffFile(str(tiff_path)) as tf:
        page0 = tf.pages[0]
        shape = page0.shape
        height, width = int(shape[-2]), int(shape[-1])

        # Pixelsize-Detection (bestehende Logik hier einbauen)
        px_um, px_src = _detect_pixelsize_from_tifffile(tf, tiff_path, width, height)

        return TiffMetadata(
            path=tiff_path,
            width=width,
            height=height,
            shape=shape,
            pixelsize_um=px_um,
            pixelsize_src=px_src,
            imagej_metadata=getattr(tf, 'imagej_metadata', None),
            ome_metadata=getattr(tf, 'ome_metadata', None)
        )

def _detect_pixelsize_from_tifffile(tf: tfi.TiffFile, tiff_path: Path,
                                     width: int, height: int) -> Tuple[Optional[float], str]:
    """Extrahiert Pixelsize aus bereits geöffnetem TiffFile (keine neue Datei-I/O!)"""
    # Alle bisherigen Detection-Steps hier (ImageJ, ResolutionTags, OME, etc.)
    # OHNE nochmal imread() oder TiffFile() zu callen
    # ... (bestehende Logik von detect_pixelsize_um_per_px kopieren)
    pass

def process_one_image_optimized(inp: Path, out: Path) -> dict:
    """Optimierte Version - TIFF nur einmal laden"""

    # 1. Metadaten einmal extrahieren
    metadata = extract_tiff_metadata(inp)
    px_um = metadata.pixelsize_um or pixelsize_fallback_um_per_px
    px_src = metadata.pixelsize_src

    LOG.info(f"   - pixelsize: {px_um:.6f} µm/px ({px_src})")

    # 2. Structure-Analyse (lädt TIFF intern - unvermeidbar)
    sarc = Structure(str(inp), pixelsize=px_um)
    # ... (Rest wie bisher)

    # 3. Für Overlay: Nutze bereits geladenes Array oder cache es
    # Option A: Array in metadata cachen (Memory-intensiv!)
    # Option B: Array in process_one_image einmal laden und weitergeben

    # ... (Rest der Funktion)
```

### Alternative: Array-Caching (falls Speicher ausreicht)

```python
@dataclass
class TiffMetadata:
    # ... (wie oben)
    array_2d: Optional[np.ndarray] = None  # Cache des Bildarrays

def extract_tiff_metadata(tiff_path: Path, cache_array: bool = False) -> TiffMetadata:
    with tfi.TiffFile(str(tiff_path)) as tf:
        # ... (Metadaten wie oben)

        array_2d = None
        if cache_array:
            arr = tf.asarray()
            array_2d = _ensure_2d(arr)

        return TiffMetadata(..., array_2d=array_2d)

# Dann in build_final_zmask_exact:
def build_final_zmask_exact_optimized(base_array_2d: np.ndarray,
                                      zmask_path: Path, ...):
    """Nutzt bereits geladenes Array statt nochmal zu laden"""
    # base = imread(str(original_path))  # ← ENTFERNT!
    base2d = base_array_2d  # ← Array wird übergeben
    # ... (Rest wie bisher)
```

---

## 🔧 Priorität 3: Regex Pre-Kompilierung (5-15% Speedup)

### Problem: Regex wird in Schleifen kompiliert
```python
# Zeile 183 - in detect_pixelsize_via_fiji(), für JEDES Bild!
m_unit = re.search(r"\b(?:unit|Unit)\s*[:=]\s*([^\s;,\n\r]+)", t)
m_pw = re.search(r"\b(?:pixelWidth|pixel[_\s]*width)\s*[:=]\s*([0-9.+\-eE]+)\s*([A-Za-zµμ]*)", t)
# etc.
```

### Optimierte Lösung:

```python
# Am Anfang der Datei (nach Imports):
# --------- PRE-COMPILED REGEX PATTERNS ---------
REGEX_UNIT = re.compile(r"\b(?:unit|Unit|spaceunits)\s*[:=]\s*([^\s;,\n\r]+)")
REGEX_PIXELWIDTH = re.compile(
    r"\b(?:pixelWidth|pixel[_\s]*width|x[_\s]*scale|x[_\s]*spacing)\s*[:=]\s*"
    r"([0-9.+\-eE]+)\s*([A-Za-zµμ]*)"
)
REGEX_PIXELHEIGHT = re.compile(
    r"\b(?:pixelHeight|pixel[_\s]*height|y[_\s]*scale|y[_\s]*spacing)\s*[:=]\s*"
    r"([0-9.+\-eE]+)\s*([A-Za-zµμ]*)"
)
REGEX_XRES = re.compile(r"\b(?:x[_\s]*resolution)\s*[:=]\s*([0-9.+\-eE]+)")
REGEX_YRES = re.compile(r"\b(?:y[_\s]*resolution)\s*[:=]\s*([0-9.+\-eE]+)")
REGEX_LIF_FILENAME = re.compile(r"(.+?\.lif)\s*-\s*", re.I)

# Dann in den Funktionen:
def detect_pixelsize_via_fiji(tiff_path: Path):
    # ...
    if info:
        t = info.replace(",", ".")
        m_unit = REGEX_UNIT.search(t)  # ← Nutzt pre-kompiliertes Pattern
        m_pw = REGEX_PIXELWIDTH.search(t)
        m_ph = REGEX_PIXELHEIGHT.search(t)
        # ...

def detect_pixelsize_from_neighbor_lif(tiff_path: Path, width: int, height: int):
    # Zeile 214:
    m = REGEX_LIF_FILENAME.search(tiff_path.name)  # ← Pre-kompiliert
    # ...
```

**Erwarteter Speedup:** 5-15% bei vielen Dateien, da re.compile() nur einmal läuft.

---

## 🎨 Priorität 4: In-Place Array-Operationen (10-20% Speedup)

### Problem: Unnötige Array-Kopien
```python
def _to_uint8_grayscale(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)  # Kopie 1
    amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
    # ...
    a = (a - amin) / (amax - amin)  # Kopie 2 (neues Array)
    return (np.clip(a, 0, 1) * 255.0).astype(np.uint8)  # Kopie 3
```

### Optimierte Version:

```python
def _to_uint8_grayscale_optimized(arr: np.ndarray) -> np.ndarray:
    """Minimiert Array-Kopien durch in-place Operationen"""
    # Direkt als float32 view (falls möglich) oder einmalige Kopie
    if arr.dtype == np.float32:
        a = arr.copy()  # Nur wenn wir Original nicht ändern dürfen
    else:
        a = arr.astype(np.float32, copy=False)  # copy=False wenn möglich

    # Nan-Handling
    amin = np.nanmin(a)
    amax = np.nanmax(a)

    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
        return np.zeros(a.shape, dtype=np.uint8)

    # In-place Operationen
    a -= amin          # In-place Subtraktion
    a /= (amax - amin) # In-place Division
    np.clip(a, 0, 1, out=a)  # In-place Clipping
    a *= 255.0         # In-place Multiplikation

    return a.astype(np.uint8)  # Finale Konvertierung (unvermeidbar)
```

### Weitere Array-Optimierung: _resize_mask_to()

```python
def _resize_mask_to_optimized(mask: np.ndarray, target_hw):
    """Vermeidet unnötige Resize-Operation"""
    if mask.shape == tuple(target_hw):
        return mask  # Kein Resize nötig!

    from skimage.transform import resize
    # Für Masken: order=0 (nearest neighbor) ist schneller als Interpolation
    return resize(
        mask.astype(np.uint8, copy=False),  # copy=False
        target_hw,
        order=0,
        preserve_range=True,
        anti_aliasing=False  # Für Masken nicht nötig
    ).astype(bool)  # Finale Konvertierung
```

---

## 🔤 Priorität 5: String-Operations optimieren (2-5% Speedup)

### Problem: Mehrfache String-Kopien in _unit_to_um()
```python
def _unit_to_um(unit: str) -> Optional[float]:
    u = (unit or "").strip()      # Kopie 1
    u = u.replace("μ", "µ")       # Kopie 2
    u = u.lower()                 # Kopie 3
    if u in {"µm","um", ...}:
        return 1.0
    # ... (viele if-Bedingungen)
```

### Optimierte Version mit Lookup-Dictionary:

```python
# Am Anfang der Datei: Pre-definiertes Mapping
_UNIT_TO_UM_MAP = {
    # Mikrometer (verschiedene Schreibweisen)
    "µm": 1.0, "um": 1.0, "micron": 1.0, "microns": 1.0,
    "micrometer": 1.0, "micrometers": 1.0, "micrometre": 1.0, "micrometres": 1.0,
    "µm/px": 1.0, "um/px": 1.0, "μm": 1.0,  # auch μ (Unicode MICRO SIGN)

    # Nanometer
    "nm": 1e-3, "nanometer": 1e-3, "nanometers": 1e-3,
    "nanometre": 1e-3, "nanometres": 1e-3,

    # Millimeter
    "mm": 1e3, "millimeter": 1e3, "millimeters": 1e3,
    "millimetre": 1e3, "millimetres": 1e3,

    # Zentimeter
    "cm": 1e4, "centimeter": 1e4, "centimeters": 1e4,
    "centimetre": 1e4, "centimetres": 1e4,

    # Inch
    "inch": 25400.0, "in": 25400.0,
}

def _unit_to_um_optimized(unit: str) -> Optional[float]:
    """Optimierte Version mit single lookup"""
    if not unit:
        return None

    # Normalisierung in einem Schritt
    normalized = unit.strip().replace("μ", "µ").lower()

    # Direktes Lookup (O(1) statt mehrere if-Checks)
    return _UNIT_TO_UM_MAP.get(normalized, None)
```

**Speedup:** ~2-5% da weniger String-Operationen und O(1) Dictionary-Lookup.

---

## 📋 Priorität 6: Effizientere Dateisammlung

### Problem: Ineffiziente Set-Konvertierung
```python
files = []
for pat in ["*.tif","*.tiff","*.TIF","*.TIFF"]:
    found = (indir.rglob(pat) if recurse else indir.glob(pat))
    for p in found:
        try:
            files.append(p.resolve())
        except Exception:
            continue
files = sorted({p for p in files})  # Set-Konvertierung teuer bei vielen Dateien
```

### Optimierte Version:

```python
def collect_tiff_files_optimized(indir: Path, recurse: bool = False) -> list:
    """Sammelt TIFF-Dateien effizient ohne redundante Set-Konvertierung"""
    seen = set()
    files = []

    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    glob_func = indir.rglob if recurse else indir.glob

    for pat in patterns:
        for p in glob_func(pat):
            try:
                resolved = p.resolve()
                # Duplikat-Check während der Sammlung (effizienter)
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(resolved)
            except Exception:
                continue

    # Sortierung nur einmal
    files.sort()
    return files
```

**Alternative (noch schneller bei case-insensitive Filesystemen):**

```python
def collect_tiff_files_fast(indir: Path, recurse: bool = False) -> list:
    """Nutzt glob mit case-insensitive Pattern (wenn Filesystem unterstützt)"""
    glob_func = indir.rglob if recurse else indir.glob

    try:
        # Versuche case-insensitive glob (Python 3.12+)
        files = list(glob_func("*.[Tt][Ii][Ff]"))
        files.extend(glob_func("*.[Tt][Ii][Ff][Ff]"))
    except Exception:
        # Fallback für ältere Python-Versionen
        files = []
        for pat in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
            files.extend(glob_func(pat))

    # Duplikate entfernen + sortieren
    return sorted(set(p.resolve() for p in files))
```

---

## 🧪 Priorität 7: Memory-Optimierung für Overlay-Generierung

### Problem: Mehrfaches Laden großer Arrays

```python
def save_zband_overlay_with_mask(...):
    base = imread(str(original_path))  # Voller TIFF-Load
    base2d = _ensure_2d(base)
    base8 = _to_uint8_grayscale(base2d)
    rgb = np.stack([base8, base8, base8], axis=-1).astype(np.float32)  # 3x Speicher!
    # ...
```

### Optimierte Version (speicherschonender):

```python
def save_zband_overlay_optimized(original_path: Path, final_mask_bool: np.ndarray,
                                  out_path: Path, color=(0,255,0), alpha=0.55) -> None:
    """Memory-optimierte Version für Overlay-Generierung"""
    from tifffile import imread, imwrite

    # 1. Bild einmal laden und direkt zu uint8 konvertieren
    base = imread(str(original_path))
    base2d = _ensure_2d(base)
    base8 = _to_uint8_grayscale_optimized(base2d)  # Nutzt optimierte Version

    # 2. RGB Stack direkt als uint8 (nicht float32!)
    rgb = np.stack([base8, base8, base8], axis=-1)  # uint8, nicht float32

    # 3. Overlay nur auf Maske anwenden (spart Operationen)
    if np.any(final_mask_bool):
        color_arr = np.array(color, dtype=np.uint8)

        # In-place Blending mit integer arithmetic (schneller als float)
        mask_idx = np.where(final_mask_bool)
        for c in range(3):
            # Integer-basiertes Alpha-Blending: (1-α)*base + α*color
            rgb[mask_idx[0], mask_idx[1], c] = (
                ((100 - int(alpha*100)) * rgb[mask_idx[0], mask_idx[1], c] +
                 int(alpha*100) * color_arr[c]) // 100
            ).astype(np.uint8)

    # 4. Direkt speichern (schon uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(out_path), rgb, photometric='rgb', compression='deflate')  # + Kompression
```

---

## 📊 Zusammenfassung Performance-Gewinn

| Optimierung | Speedup | Memory-Reduktion | Implementierungsaufwand |
|-------------|---------|------------------|-------------------------|
| **Multiprocessing** | **4-8x** | - | Mittel (Fiji-Handling!) |
| **TIFF-Caching** | 20-40% | - | Mittel-Hoch |
| **Regex Pre-Compile** | 5-15% | - | Sehr gering |
| **In-Place Arrays** | 10-20% | 30-50% | Gering |
| **String-Optimierung** | 2-5% | - | Sehr gering |
| **Dateisammlung** | 1-3% | - | Sehr gering |
| **Overlay-Optimierung** | 5-10% | 50% | Gering |

**Gesamt-Speedup bei Kombination aller Optimierungen:** ~5-10x (hauptsächlich durch Multiprocessing)

---

## 🔧 Implementierungs-Roadmap

### Phase 1: Quick Wins (1-2 Stunden)
1. ✅ Regex pre-kompilieren
2. ✅ String-Operations optimieren
3. ✅ Dateisammlung verbessern

### Phase 2: Medium Impact (4-6 Stunden)
4. ✅ In-place Array-Operationen
5. ✅ Overlay-Generierung optimieren

### Phase 3: High Impact (8-12 Stunden)
6. ✅ TIFF-Metadaten cachen
7. ✅ Multiprocessing implementieren (mit Fiji-Workarounds!)

### Phase 4: Testing & Validation
- Benchmark alter vs. neuer Code
- Memory-Profiling
- Edge-Case Tests

---

## ⚠️ Wichtige Warnings

### Multiprocessing + Fiji:
```python
# PROBLEM: Fiji/PyImageJ ist NICHT prozess-sicher!
# LÖSUNG 1: Fiji pro Worker initialisieren
# LÖSUNG 2: Fiji global deaktivieren bei Multiprocessing:
if use_multiprocessing and enable_fiji_via_pyimagej:
    LOG.warning("Fiji wird bei Multiprocessing deaktiviert (nicht thread-safe)")
    enable_fiji_via_pyimagej = False
```

### Memory bei Array-Caching:
```python
# Bei großen TIFFs (>100 MB) kann Caching problematisch sein
# → Nur Metadaten cachen, nicht die Arrays!
```

### Kompatibilität:
- Multiprocessing benötigt Python 3.7+
- Pre-compiled Regex kompatibel mit allen Python-Versionen
- In-place Array-Ops benötigen NumPy 1.13+

---

Möchtest du, dass ich eine dieser Optimierungen direkt in den Code implementiere?
