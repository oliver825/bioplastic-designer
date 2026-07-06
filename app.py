"""
Desktop GUI for the bioplastic inverse-design model.

Run this file (green Run button in PyCharm, or `python app.py`).
Needs inverse_design.py in the same folder and a data.csv to train on.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

from inverse_design import (
    MixtureModel, DEFAULT_INPUT_COLS, DEFAULT_TARGET_COLS, DEFAULT_BOUNDS,
)

# ---- palette ----------------------------------------------------------------
BG      = "#EEF1F4"
CARD    = "#FFFFFF"
BORDER  = "#DAE0E6"
TEXT    = "#1F2933"
MUTED   = "#6B7280"
ACCENT  = "#2F855A"
ACCENT_DK = "#276749"
RED     = "#C0392B"
AMBER   = "#B7791F"
GREEN   = "#2F855A"
TRACK   = "#E6E9ED"
FONT    = "Segoe UI"
MONO    = "Consolas"


class DataMeter(tk.Canvas):
    """A colour-coded x / y progress bar for data sufficiency."""
    def __init__(self, master):
        super().__init__(master, height=64, highlightthickness=0, bg=CARD)
        self._x = self._y = self._nvary = 0
        self._has = False
        self.bind("<Configure>", lambda e: self._draw())

    def set(self, x, y, nvary):
        self._x, self._y, self._nvary, self._has = x, y, nvary, True
        self._draw()

    def _draw(self):
        self.delete("all")
        w = self.winfo_width() or 560
        if not self._has:
            self.create_text(0, 32, anchor="w", text="No data loaded yet.",
                             fill=MUTED, font=(FONT, 10))
            return
        x, y, nvary = self._x, self._y, self._nvary
        frac = max(0.0, min(x / y, 1.0))
        colour = RED if frac < 0.34 else (AMBER if frac < 0.75 else GREEN)

        # numbers
        self.create_text(0, 10, anchor="w", text=f"{x} / {y} rows",
                         fill=TEXT, font=(FONT, 13, "bold"))
        if x >= y:
            note = "enough to start \u2014 keep adding for higher confidence"
        else:
            note = f"~{y - x} more before predictions start becoming usable"
        self.create_text(w, 10, anchor="e", text=note, fill=MUTED, font=(FONT, 9))

        # track + fill (rounded-ish)
        top, bot = 30, 44
        self.create_rectangle(0, top, w, bot, fill=TRACK, outline="")
        self.create_rectangle(0, top, max(2, w * frac), bot, fill=colour, outline="")
        self.create_text(0, 56, anchor="w",
                         text=f"estimate based on {nvary} material(s) that vary "
                              f"(~10 experiments each)",
                         fill=MUTED, font=(FONT, 8))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bioplastic Blend \u2014 Inverse Design")
        self.geometry("1220x700")
        self.minsize(1080, 620)
        self.configure(bg=BG)

        self.model = MixtureModel(DEFAULT_INPUT_COLS, DEFAULT_TARGET_COLS, DEFAULT_BOUNDS)
        self.trained = False
        self._init_style()
        self._build()

    # ---- styling ----
    def _init_style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("Accent.TButton", font=(FONT, 10, "bold"), foreground="white",
                    background=ACCENT, borderwidth=0, padding=(16, 9))
        s.map("Accent.TButton", background=[("active", ACCENT_DK), ("pressed", ACCENT_DK)])
        s.configure("Ghost.TButton", font=(FONT, 10), foreground=ACCENT,
                    background=CARD, bordercolor=BORDER, borderwidth=1, padding=(12, 7))
        s.map("Ghost.TButton", background=[("active", "#F0F4F1")])
        s.configure("TEntry", fieldbackground="white", bordercolor=BORDER,
                    lightcolor=BORDER, borderwidth=1, padding=6)

    def _card(self, parent, title, fill="x", expand=False):
        outer = tk.Frame(parent, bg=BORDER)  # 1px border via padding trick
        outer.pack(fill=fill, expand=expand, padx=10, pady=8)
        card = tk.Frame(outer, bg=CARD)
        card.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(card, text=title, bg=CARD, fg=ACCENT,
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=16, pady=(12, 4))
        body = tk.Frame(card, bg=CARD)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        return body

    # ---- layout ----
    def _build(self):
        # header band
        header = tk.Frame(self, bg=ACCENT)
        header.pack(fill="x")
        tk.Label(header, text="Bioplastic Blend Designer", bg=ACCENT, fg="white",
                 font=(FONT, 16, "bold")).pack(anchor="w", padx=20, pady=(14, 2))
        tk.Label(header, text="Predict a recipe from the properties you want",
                 bg=ACCENT, fg="#D7E8DE", font=(FONT, 10)).pack(anchor="w", padx=20, pady=(0, 14))

        # two columns: inputs on the left, results on the right
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, minsize=470, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)
        left = tk.Frame(content, bg=BG)
        left.grid(row=0, column=0, sticky="nsew")
        right = tk.Frame(content, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")

        # 1. Data (left)
        d = self._card(left, "1.  Data")
        row = tk.Frame(d, bg=CARD); row.pack(fill="x")
        self.path_var = tk.StringVar(value="data.csv")
        ttk.Entry(row, textvariable=self.path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse", style="Ghost.TButton",
                   command=self._browse).pack(side="left", padx=(8, 0))
        ttk.Button(row, text="Load & Train", style="Accent.TButton",
                   command=self._train).pack(side="left", padx=(8, 0))

        self.meter = DataMeter(d)
        self.meter.pack(fill="x", pady=(12, 6))

        self.status = tk.Label(d, text="No model trained yet.", bg=CARD,
                               fg=MUTED, font=(FONT, 9))
        self.status.pack(anchor="w")
        ttk.Button(d, text="Check model quality", style="Ghost.TButton",
                   command=self._quality).pack(anchor="w", pady=(8, 4))
        self.quality = tk.Label(d, text="", bg=CARD, fg=TEXT, justify="left",
                                font=(FONT, 9), wraplength=430)
        self.quality.pack(anchor="w")

        # 2. Targets (left)
        t = self._card(left, "2.  Desired properties")
        self.target_vars = {}
        for col in DEFAULT_TARGET_COLS:
            rr = tk.Frame(t, bg=CARD); rr.pack(fill="x", pady=3)
            tk.Label(rr, text=col, bg=CARD, fg=TEXT, font=(FONT, 10),
                     width=22, anchor="w").pack(side="left")
            v = tk.StringVar(); self.target_vars[col] = v
            ttk.Entry(rr, textvariable=v, width=16).pack(side="left")
        ttk.Button(t, text="Suggest recipe", style="Accent.TButton",
                   command=self._suggest).pack(anchor="w", pady=(10, 2))

        # 3. Result (right, fills the column)
        res = self._card(right, "3.  Suggested formulation", fill="both", expand=True)
        self.result = tk.Text(res, height=20, wrap="word", font=(MONO, 10),
                              relief="flat", background="#FBFCFD", foreground=TEXT,
                              padx=12, pady=10, borderwidth=0)
        self.result.pack(fill="both", expand=True)
        self.result.tag_configure("bad",  foreground=RED,   font=(FONT, 10, "bold"))
        self.result.tag_configure("warn", foreground=AMBER, font=(FONT, 10, "bold"))
        self.result.tag_configure("good", foreground=GREEN, font=(FONT, 10, "bold"))
        self.result.tag_configure("head", foreground=TEXT, font=(FONT, 10, "bold"))
        self.result.tag_configure("muted", foreground=MUTED)
        self.result.configure(state="disabled")

    # ---- helpers ----
    def _browse(self):
        p = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self.path_var.set(p)

    def _set_status(self, text, colour=MUTED):
        self.status.config(text=text, fg=colour); self.update_idletasks()

    def _read_csv(self, path):
        df = pd.read_csv(path)
        if df.shape[1] == 1:          # likely semicolon-separated (Dutch Excel)
            df = pd.read_csv(path, sep=";", decimal=",")
        return df

    # ---- actions ----
    def _train(self):
        path = self.path_var.get().strip()
        if not os.path.exists(path):
            messagebox.showerror("File not found", f"Could not find:\n{path}")
            return
        try:
            self._set_status("Reading data\u2026")
            df = self._read_csv(path)
            self._set_status("Training models\u2026 (a few seconds)")
            self.model.fit(df)
            self.trained = True
            nvary, target = self.model.recommended_rows()
            self.meter.set(self.model.n_rows, target, nvary)
            self._set_status(f"Trained on {self.model.n_rows} rows. Ready.", ACCENT_DK)
            self.quality.config(text="")
        except Exception as e:
            self.trained = False
            self._set_status("Training failed.", RED)
            messagebox.showerror("Could not train", str(e))

    def _quality(self):
        if not self.trained:
            messagebox.showwarning("No model yet", "Load & train a dataset first.")
            return
        try:
            self._set_status("Checking accuracy (leave-one-out)\u2026")
            rep = self.model.cv_report()
            lines = ["Leave-one-out accuracy  (R\u00b2 near 1 = good, near 0 = weak,"
                     " negative = worse than guessing the average):"]
            for col, d in rep.items():
                lines.append(f"   {col}:  R\u00b2 = {d['r2']:.2f}   typical error \u00b1 {d['rmse']:.2f}")
            cov = self.model.data_coverage()
            constant = [m for m, dd in cov.items() if not dd['varies']]
            if constant:
                lines.append("")
                lines.append("Constant in every row (model learns nothing about these):")
                for m in constant:
                    lines.append(f"   \u2022 {m}")
            self.quality.config(text="\n".join(lines))
            self._set_status(f"Trained on {self.model.n_rows} rows. Ready.", ACCENT_DK)
        except Exception as e:
            self._set_status("Quality check failed.", RED)
            messagebox.showerror("Error", str(e))

    def _suggest(self):
        if not self.trained:
            messagebox.showwarning("No model yet", "Load & train a dataset first.")
            return
        targets = {}
        for col, var in self.target_vars.items():
            raw = var.get().strip().replace(",", ".")
            if raw == "":
                messagebox.showwarning("Missing value", f"Enter a target for: {col}")
                return
            try:
                targets[col] = float(raw)
            except ValueError:
                messagebox.showerror("Invalid number", f"'{var.get()}' isn't a number ({col}).")
                return
        try:
            self._set_status("Searching for the best recipe\u2026")
            out = self.model.suggest(targets)
            self._render(out, targets)
            self._set_status(f"Trained on {self.model.n_rows} rows. Ready.", ACCENT_DK)
        except Exception as e:
            self._set_status("Search failed.", RED)
            messagebox.showerror("Error", str(e))

    # ---- result rendering ----
    def _verdict(self, out, targets):
        target_outside = too_uncertain = impossible = False
        for col in out['predicted']:
            m, sgm = out['predicted'][col], out['std'][col]
            lo, hi = m - 1.96 * sgm, m + 1.96 * sgm
            if not (lo <= targets[col] <= hi):
                target_outside = True
            if 1.96 * sgm >= out['data_std'].get(col, 1.0):
                too_uncertain = True
            if lo < 0:
                impossible = True
        if too_uncertain or impossible:
            tail = (", and some ranges dip below zero into the physically impossible"
                    if impossible else "")
            return "bad", [
                "VERDICT: do NOT trust this recipe.",
                f"The 95% ranges are as wide as the spread of your own data{tail}.",
                "The model is effectively making a complete guess. More varied data is needed."
            ]
        if target_outside:
            return "warn", [
                "NOTE: a target falls outside the model's 95% range. These materials",
                "may be unable to reach it, or the model is extrapolating. Verify in the lab.",
            ]
        return "good", ["Targets are within range and the model is reasonably confident."]

    def _render(self, out, targets):
        self.result.configure(state="normal")
        self.result.delete("1.0", "end")

        # Recipe as a two-column table (fewer rows, uses horizontal space)
        self.result.insert("end", "SUGGESTED RECIPE  (fraction of total formulation)\n", "head")
        items = list(out['recipe'].items())
        for k in range(0, len(items), 2):
            name_l, frac_l = items[k]
            cell = f"  {name_l:<32}{frac_l * 100:>7.2f} %"
            if k + 1 < len(items):
                name_r, frac_r = items[k + 1]
                cell += f"     {name_r:<32}{frac_r * 100:>7.2f} %"
            self.result.insert("end", cell + "\n")

        # Properties: one line each
        self.result.insert("end",
            "\nPREDICTED PROPERTIES  (target \u2192 predicted, 95% range)\n", "head")
        for col in out['predicted']:
            m, sgm, t = out['predicted'][col], out['std'][col], targets[col]
            lo, hi = m - 1.96 * sgm, m + 1.96 * sgm
            self.result.insert("end",
                f"  {col:<22} target {t:g}  \u2192  predicted {m:.2f}"
                f"   (95%: {lo:.2f} to {hi:.2f})\n")

        # Verdict underneath
        level, vlines = self._verdict(out, targets)
        self.result.insert("end", "\n")
        for ln in vlines:
            self.result.insert("end", ln + "\n", level)

        # Brief explanation of the 95% range
        self.result.insert("end",
            "\nThe 95% range is a confidence interval. This is the band that the model is 95% sure that the true value falls in.\n"
            "For example, a 95% range of 20 to 25 for Tensile strength means that the model believes there is a 95% chance that the value of Tensile strength created by the recipe would lie between 20 and 25 MPa.\n\n"
            "The model's predicted value is the midpoint of this range.\n"
            "Narrow = confident (recipe is near your data)\nWide = unsure (extrapolating).\n",
            "muted")
        self.result.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()