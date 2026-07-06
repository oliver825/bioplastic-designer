"""
Inverse design for bioplastic / PHA blend formulations.

A Gaussian Process forward model (one per property) learns how composition maps
to properties, INCLUDING non-additive interaction effects -- it does not assume
the relationship is linear or additive. An optimizer then searches composition
space for the recipe whose predicted properties best match a target.

The kernel is configured to stay smooth rather than collapse to memorising the
training points (the failure mode you hit with very few rows), so the inverse
search always has a usable gradient to follow. Predictions come with an
uncertainty estimate, and cv_report() measures real predictive accuracy so you
can see the model improve as you add data.
"""

import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning


# ---- Configuration (single source of truth, also imported by the GUI) --------

DEFAULT_INPUT_COLS = [
    'PLA',
    'P(3HB-co-4HB)',
    'P(3HB)',
    'mcl-PHA',
    'PEG (Plasticizer)',
    'Nucleating Agent (Boron Nitride)',
    'Filler (CaCO3)',
]

DEFAULT_TARGET_COLS = ['Tensile Strength (MPa)', 'Elongation at Break (%)', 'Modulus 100 (MPa)']

# (min, max) fraction of the total formulation for each material.
DEFAULT_BOUNDS = {
    'PLA':                              (0.0, 1.0),
    'P(3HB-co-4HB)':                    (0.0, 1.0),
    'P(3HB)':                           (0.0, 1.0),
    'mcl-PHA':                          (0.0, 1.0),
    'PEG (Plasticizer)':                (0.0, 0.20),
    'Nucleating Agent (Boron Nitride)': (0.0, 0.02),
    'Filler (CaCO3)':                   (0.0, 0.30),
}


def _make_kernel():
    # Matern(nu=2.5): smooth, can represent interactions (it's not additive).
    # length_scale lower bound is kept at 0.1 (in composition units, 0-1) so the
    # model CANNOT shrink it to ~0 and collapse into memorise-and-average.
    # WhiteKernel gives a noise floor so scatter is blamed on measurement noise
    # rather than on a spiky function.
    return (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
    )


def _repair(x, lo, hi):
    """Make a recipe sum to exactly 1 while keeping every material in [lo, hi]."""
    x = np.clip(np.asarray(x, dtype=float), lo, hi)
    for _ in range(60):
        residual = 1.0 - x.sum()
        if abs(residual) < 1e-12:
            break
        room = (hi - x) if residual > 0 else (x - lo)
        total = room.sum()
        if total < 1e-12:
            break
        x = np.clip(x + np.sign(residual) * room * (abs(residual) / total), lo, hi)
    return x


class MixtureModel:
    def __init__(self, input_cols=None, target_cols=None, material_bounds=None):
        self.input_cols = list(input_cols or DEFAULT_INPUT_COLS)
        self.target_cols = list(target_cols or DEFAULT_TARGET_COLS)
        self.material_bounds = dict(material_bounds or DEFAULT_BOUNDS)
        self.gp_models = None
        self.target_scales = None
        self.n_rows = 0
        self._X = None
        self._y = None

    # ---- training ----
    def fit(self, df):
        missing = [c for c in self.input_cols + self.target_cols if c not in df.columns]
        if missing:
            raise ValueError("CSV is missing these columns:\n  " + "\n  ".join(missing))

        X = df[self.input_cols].to_numpy(dtype=float)
        y = df[self.target_cols].to_numpy(dtype=float)
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError(
                "Data contains blank/NaN cells. Every cell needs a number "
                "(use 0 for a material that isn't in a blend)."
            )

        self._X, self._y = X, y
        self.n_rows = X.shape[0]
        self.target_scales = y.std(axis=0)
        self.target_scales[self.target_scales == 0] = 1.0

        self.gp_models = []
        for i in range(len(self.target_cols)):
            gp = GaussianProcessRegressor(
                kernel=_make_kernel(), normalize_y=True,
                n_restarts_optimizer=10, random_state=42,
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                gp.fit(X, y[:, i])
            self.gp_models.append(gp)
        return self

    # ---- forward prediction ----
    def predict(self, ratios):
        R = np.atleast_2d(np.asarray(ratios, dtype=float))
        means = np.zeros((R.shape[0], len(self.target_cols)))
        stds = np.zeros_like(means)
        for i, gp in enumerate(self.gp_models):
            m, s = gp.predict(R, return_std=True)
            means[:, i] = m
            stds[:, i] = s
        return means, stds

    def _objective(self, ratios, target):
        means, _ = self.predict(ratios)
        return float(np.sum(((means[0] - target) / self.target_scales) ** 2))

    # ---- inverse design ----
    def suggest(self, targets, n_random_starts=16, seed=42):
        if isinstance(targets, dict):
            target = np.array([targets[c] for c in self.target_cols], dtype=float)
        else:
            target = np.asarray(targets, dtype=float)

        lo = np.array([self.material_bounds[m][0] for m in self.input_cols])
        hi = np.array([self.material_bounds[m][1] for m in self.input_cols])
        if lo.sum() > 1.0 + 1e-9 or hi.sum() < 1.0 - 1e-9:
            raise ValueError(
                f"Material bounds can't make a recipe summing to 1 "
                f"(lower sum {lo.sum():.3f}, upper sum {hi.sum():.3f})."
            )
        bounds = list(zip(lo, hi))
        constraints = ({'type': 'eq', 'fun': lambda r: np.sum(r) - 1.0})

        rng = np.random.default_rng(seed)
        # Starts: midpoint, random points, AND every known recipe (so the search
        # also begins from the informative data points, not just blind guesses).
        starts = [_repair((lo + hi) / 2.0, lo, hi)]
        for _ in range(n_random_starts):
            starts.append(_repair(rng.uniform(lo, hi), lo, hi))
        for row in self._X:
            starts.append(_repair(row, lo, hi))

        best = None
        for x0 in starts:
            res = minimize(self._objective, x0, args=(target,), method='SLSQP',
                           bounds=bounds, constraints=constraints)
            x = _repair(res.x, lo, hi)
            val = self._objective(x, target)
            if best is None or val < best[0]:
                best = (val, x)

        recipe = best[1]
        means, stds = self.predict(recipe)
        return {
            'recipe':    dict(zip(self.input_cols, recipe)),
            'predicted': dict(zip(self.target_cols, means[0])),
            'std':       dict(zip(self.target_cols, stds[0])),
            'data_std':  dict(zip(self.target_cols, self.target_scales)),
            'error':     best[0],
        }

    def data_coverage(self):
        """For each material, its range in the data and whether it varies at
        all. A material that never changes teaches the model nothing."""
        out = {}
        for i, m in enumerate(self.input_cols):
            col = self._X[:, i]
            out[m] = {
                'min': float(col.min()),
                'max': float(col.max()),
                'varies': bool(col.max() - col.min() > 1e-9),
            }
        return out

    def recommended_rows(self):
        """Rough heuristic for how many experiments the model wants before it
        starts giving meaningful predictions: ~10 per material that varies.
        Returns (n_varying_materials, target_rows). Not a guarantee -- variety
        of the blends matters as much as the count."""
        n_vary = max(1, sum(d['varies'] for d in self.data_coverage().values()))
        target = max(25, 10 * n_vary)
        return n_vary, target

    # ---- honest accuracy check ----
    def cv_report(self):
        """Leave-one-out cross-validation per property: R2 and RMSE.
        R2 near 1 = good; near 0 = no better than guessing the average;
        negative = worse than the average (usually too little data)."""
        n = self._X.shape[0]
        report = {}
        for i, col in enumerate(self.target_cols):
            preds = np.zeros(n)
            for j in range(n):
                mask = np.arange(n) != j
                gp = GaussianProcessRegressor(
                    kernel=_make_kernel(), normalize_y=True,
                    n_restarts_optimizer=3, random_state=42,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    gp.fit(self._X[mask], self._y[mask, i])
                preds[j] = gp.predict(self._X[j:j + 1])[0]
            actual = self._y[:, i]
            ss_res = np.sum((actual - preds) ** 2)
            ss_tot = np.sum((actual - actual.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            rmse = float(np.sqrt(np.mean((actual - preds) ** 2)))
            report[col] = {'r2': float(r2), 'rmse': rmse}
        return report


if __name__ == '__main__':
    model = MixtureModel()
    df = pd.read_csv('data.csv')
    model.fit(df)
    print(f"Trained on {model.n_rows} rows.\n")

    print("Leave-one-out accuracy:")
    for col, d in model.cv_report().items():
        print(f"  {col}: R2={d['r2']:.2f}, RMSE={d['rmse']:.2f}")

    out = model.suggest({'Tensile Strength (MPa)': 41, 'Elongation at Break (%)': 250, 'Modulus 100 (MPa)': 1103})
    print("\nSuggested recipe:")
    for mat, frac in out['recipe'].items():
        print(f"  {mat}: {frac:.2%}")
    print("\nPredicted:")
    for col in out['predicted']:
        m, s = out['predicted'][col], out['std'][col]
        print(f"  {col}: {m:.2f}  (95%: {m - 1.96 * s:.2f} to {m + 1.96 * s:.2f})")
