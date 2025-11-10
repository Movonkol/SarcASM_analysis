# SarcAsM batch v8.2 – Installation & Usage Guide (VS Code friendly)

This script runs the **SArcasM batch analysis** on multiple TIFF images and writes the main metrics to a CSV file.  
It is designed so that biologists can use it with **Python + VS Code** without needing deep programming knowledge.

---

## 1. What you do in practice

1. Install **Python** and **VS Code**.  
2. Create a folder for the project (e.g. `SArcasm/`).  
3. In VS Code, create a file `SArcasm.py` and **paste the script code** from GitHub.  
4. In VS Code, open the built-in **Terminal** and run a few `pip install` commands.  
5. Adjust the **USER SETTINGS** at the top of `SArcasm.py` (input folder, output folder, pixel size).  
6. Click the **green “Run” button** in VS Code to start the analysis.

You do **not** need to type `python SArcasm.py` manually to run it (only for installing packages).

---

## 2. Requirements

### 2.1 Operating system

- Windows 10 / 11  
- macOS or Linux will also work, but examples below use Windows paths.

### 2.2 Software

1. **Python 3.9+**
   - Download from the official Python website.
   - During installation on Windows, make sure to tick:

     **“Add Python to PATH”**
   - Check in a terminal (Command Prompt or PowerShell):

     ```bash
     python --version
     ```

2. **Visual Studio Code (VS Code)**
   - Download from the official VS Code website.
   - Start VS Code.
   - Install the **Python extension**:
     - Click the **Extensions** icon (four small squares) on the left.
     - Search for **“Python”** (by Microsoft) and install it.

3. (Optional) **Java + Fiji/PyImageJ**
   - Only needed if you want **automatic µm/px detection** using PyImageJ and BioFormats.
   - Install a Java JDK (e.g. Temurin JDK 8, 11, or 17).
   - You can still use the script without this and just define a fixed pixel size.

---

## 3. Getting the code into VS Code

You have two options:

### Option A – Clone the repo

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_FOLDER>
```

Open this folder in VS Code.

### Option B – Copy & paste from GitHub (recommended for non-coders)

1. Create a folder on your computer, for example:

   `C:\Users\YourName\Documents\SArcasm`

2. Start **VS Code**.
3. Go to **File → Open Folder…** and select the `SArcasm` folder.
4. In the VS Code file explorer (left side), click **New File** and name it:

   `SArcasm.py`

5. On GitHub, open the script file, select the entire Python code and **copy** it.
6. Paste it into `SArcasm.py` in VS Code.
7. Save the file (`Ctrl + S`).

---

## 4. Setting up the Python environment (VS Code Terminal)

You only need the terminal for **installing packages**, not for running the script.

### 4.1 Open the VS Code terminal

1. In VS Code, go to:

   **Terminal → New Terminal**

2. A terminal opens at the bottom.  
   - On Windows this is usually **PowerShell** or **Command Prompt**.  
   - Sometimes you might see **bash** (e.g. Git Bash). That is also fine.

Check that the terminal is in the project folder. You should see a line like:

```text
C:\Users\YourName\Documents\SArcasm>
```

If not, change directory:

```bash
cd "C:\Users\YourName\Documents\SArcasm"
```

### 4.2 (Optional) Create a virtual environment

This keeps your dependencies clean:

```bash
python -m venv .venv
```

Activate it:

- PowerShell:

  ```bash
  .venv\Scripts\Activate.ps1
  ```

- Command Prompt (cmd):

  ```bash
  .venv\Scripts\activate.bat
  ```

On macOS/Linux:

```bash
source .venv/bin/activate
```

You should now see `(.venv)` at the beginning of the terminal line.

### 4.3 Install required Python packages

Run in the **VS Code terminal**:

```bash
pip install sarc-asm tifffile scikit-image
```

If you want **Fiji/PyImageJ** support (automatic pixel size from LIF/TIFF via BioFormats):

```bash
pip install pyimagej scyjava
```

If these `pip` commands run without errors, you’re ready to configure and run the script.

---

## 5. Folder structure for your data

Use a simple structure like this:

```text
SArcasm/
  SArcasm.py
  input/   → your .tif / .tiff images
  output/  → will receive CSV and overlays
```

Create the `input` and `output` folders manually in your OS or directly in VS Code.

---

## 6. Adjust the USER SETTINGS in `SArcasm.py`

At the top of the script there is a block called **USER SETTINGS**, for example:

```python
# ---------------- USER SETTINGS ----------------
input_dir = r"C:\Users\YourName\Documents\SArcasm\input"

pixelsize_fallback_um_per_px = 0.14017
auto_pixelsize = True

pixelsize_by_prefix: Dict[str, float] = {
    # "ExamplePrefix_": 0.07,
}

enable_fiji_via_pyimagej = True
fiji_maven_coord = 'sc.fiji:fiji:2.9.0'
```

The most important things to edit:

1. **Input folder with your TIFF images**

   ```python
   input_dir = r"C:\Users\YourName\Documents\SArcasm\input"
   ```

   Use an absolute path and keep the leading `r` (raw string) in front of the quotes on Windows.

2. **Output folder (for CSV + overlays)**

   Somewhere further down, you will find:

   ```python
   out_dir = r"C:\Users\YourName\Documents\SArcasm\output"
   ```

   Adjust it to your preferred output folder. The folder should exist.

3. **Pixel size (µm/px)**

   - If you want the script to **read pixel size from image metadata / BioFormats**:

     ```python
     auto_pixelsize = True
     enable_fiji_via_pyimagej = True
     ```

   - If you want to use a **fixed pixel size** and **no Fiji**:

     ```python
     auto_pixelsize = False
     pixelsize_fallback_um_per_px = 0.14  # set your value here
     enable_fiji_via_pyimagej = False
     ```

4. **Optional: custom pixel size by filename prefix**

   Example:

   ```python
   pixelsize_by_prefix = {
       "EHTM_": 0.0707,
       "Sample2_": 0.1415,
   }
   ```

   If a filename starts with one of these prefixes, that pixel size will be used.

For basic use, you can keep the default values and only change:

- `input_dir`
- `out_dir`
- `auto_pixelsize` / `pixelsize_fallback_um_per_px`
- `enable_fiji_via_pyimagej`

---

## 7. Running the script in VS Code (without typing `python SArcasm.py`)

1. Make sure `SArcasm.py` is open in the editor.
2. At the **top right** of the editor, click the **green triangle button** (Run Python file).  
   – or –  
   Go to **Run → Run Without Debugging** (`Ctrl + F5`).
3. VS Code will run the script using the selected Python interpreter.
   - If needed, select the interpreter (bottom-right blue bar → click Python version → choose the `.venv` or system Python).

You will see the script’s log output in the integrated terminal or in the “Python”/“Output” panel in VS Code.  
The script processes all `.tif` / `.tiff` files in `input_dir` and writes results to `out_dir`.

Typical outputs:

- A `.csv` file with the measured metrics.
- Optional overlay images and masks, depending on the overlay settings.

---

## 8. Troubleshooting

**Error: `ModuleNotFoundError: No module named 'sarc_asm'`**  
→ The `sarc-asm` package is missing.

```bash
pip install sarc-asm
```

**Error: `python is not recognized as an internal or external command` (Windows)**  
→ Python is not in PATH or not installed correctly. Reinstall Python and tick **“Add Python to PATH”**.

**PyImageJ/Fiji errors, but you don’t care about automatic pixel size**  
→ Disable Fiji & auto pixel size in `SArcasm.py`:

```python
auto_pixelsize = False
enable_fiji_via_pyimagej = False
pixelsize_fallback_um_per_px = 0.14  # or your value
```

After that, the script will just use the fixed fallback pixel size.

---

You can drop this file directly into your repository as `README.md` and adapt paths or parameter descriptions as needed.
