## Download and run

No Python needed — just download, unzip, and open. Get the file for your system from the [Releases page](../../releases).

**Windows**
1. Download `BioplasticDesigner-Windows.zip`.
2. Unzip it, open the folder, and double-click `BioplasticDesigner.exe`.

**macOS**
1. Download `BioplasticDesigner-macOS.zip`.
2. Unzip it and drag `BioplasticDesigner.app` into your Applications folder.
3. The first time you open it, **right-click the app → Open** (not a normal double-click). macOS shows an "unidentified developer" warning because the app isn't signed with a paid Apple certificate — clicking Open once bypasses it permanently. This is expected and safe.

---

## How to use it

1. **Load & Train** — pick your `data.csv` and click Load & Train. The bar shows how many rows you have versus roughly how many the model wants.
2. **Check model quality** (optional) — runs a leave-one-out test and reports an R² per property. Near 1 is good; near 0 or negative means there isn't enough data yet to trust predictions.
3. **Enter desired properties** and click **Suggest recipe**. You get a blend, each predicted property with a 95% confidence range, and a plain-English verdict on whether to trust it.

The 95% range is the band the model is 95% sure the true value falls in: narrow means confident (your recipe is near real data), wide means it's guessing. If the ranges are very wide, the tool will tell you outright not to trust the result — that's a sign you need more (and more varied) experiments.

---

## Files

- `app.py` — the desktop interface
- `inverse_design.py` — the model (Gaussian Processes + the inverse search)
- `requirements.txt` — Python dependencies
- `.github/workflows/build.yml` — the automated Windows/Mac build

---
