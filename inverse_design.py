"""
Inverse design for bioplastic / PHA blend formulations.

A Gaussian Process forward model (one per property) learns how composition maps
to properties, INCLUDING non-additive interaction effects -- it does not assume
the relationship is linear or additive. An optimizer then searches composition
space for the recipe whose predicted properties best match a target.

Properties listed in LOG_TARGETS are modelled on a log scale. This suits
properties that span orders of magnitude (elongation runs from ~5% to >2000%):
without it, a couple of extreme rows dominate the error and the R2 score, and
the 95% range can dip below zero into physically impossible values.

Predictions come with an uncertainty range, and cv_report() measures real
predictive accuracy so you can see the model improve as you add data.
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
]

DEFAULT_TARGET_COLS = [
    'Tensile Strength (MPa)',
    'Elongation at Break (%)',
    "Young's Modulus (MPa)",
]

# Properties modelled on a log scale. Must be strictly positive in the data.
#I chose to model only elongation on a log scale as it spans many orders and is not evenly spread
DEFAULT_LOG_TARGETS = ['Elongation at Break (%)']

# (min, max) fraction of the total formulation for each material.
DEFAULT_BOUNDS = {
    'PLA':           (0.0, 1.0),
    'P(3HB-co-4HB)': (0.0, 1.0),
    'P(3HB)':        (0.0, 1.0),
    'mcl-PHA':       (0.0, 1.0),
}

Z = 1.96  # 95% interval multiplier


def _make_kernel():
    # Matern(nu=2.5): smooth, can represent interactions (it's not additive).
    # length_scale floor of 0.1 stops the model collapsing into memorise-and-
    # average. WhiteKernel gives a noise floor so scatter is blamed on
    # measurement noise rather than on a spiky function.
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


def _fit_gp(X, y, restarts=10):
    gp = GaussianProcessRegressor(kernel=_make_kernel(), normalize_y=True,
                                  n_restarts_optimizer=restarts, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        gp.fit(X, y)
    return gp


class MixtureModel:
    def __init__(self, input_cols=None, target_cols=None, material_bounds=None,
                 log_targets=None):
        self.input_cols = list(input_cols or DEFAULT_INPUT_COLS)
        self.target_cols = list(target_cols or DEFAULT_TARGET_COLS)
        self.material_bounds = dict(material_bounds or DEFAULT_BOUNDS)
        lt = DEFAULT_LOG_TARGETS if log_targets is None else log_targets
        self.log_targets = [c for c in lt if c in self.target_cols]
        self._log_mask = np.array([c in self.log_targets for c in self.target_cols])
        self.gp_models = None
        self.target_scales = None   # in transformed space (used by the objective)
        self.data_std = None        # in original units (used by the GUI's checks)
        self.n_rows = 0
        self._X = None
        self._y = None              # original units
        self._z = None              # transformed

    # ---- log transform helpers ----
    def _fwd(self, y):
        z = np.array(y, dtype=float, copy=True)
        if self._log_mask.any():
            z[..., self._log_mask] = np.log(z[..., self._log_mask])
        return z

    def _bwd(self, z):
        y = np.array(z, dtype=float, copy=True)
        if self._log_mask.any():
            y[..., self._log_mask] = np.exp(y[..., self._log_mask])
        return y

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
        for i, c in enumerate(self.target_cols):
            if self._log_mask[i] and (y[:, i] <= 0).any():
                raise ValueError(f"'{c}' is log-scaled but contains values <= 0.")

        self._X, self._y = X, y
        self._z = self._fwd(y)
        self.n_rows = X.shape[0]

        self.target_scales = self._z.std(axis=0)
        self.target_scales[self.target_scales == 0] = 1.0
        self.data_std = y.std(axis=0)
        self.data_std[self.data_std == 0] = 1.0

        self.gp_models = [_fit_gp(X, self._z[:, i]) for i in range(len(self.target_cols))]
        return self

    # ---- forward prediction ----
    def _predict_z(self, ratios):
        """Mean and std in the (possibly log) modelling space."""
        R = np.atleast_2d(np.asarray(ratios, dtype=float))
        mu = np.zeros((R.shape[0], len(self.target_cols)))
        sd = np.zeros_like(mu)
        for i, gp in enumerate(self.gp_models):
            m, s = gp.predict(R, return_std=True)
            mu[:, i] = m
            sd[:, i] = s
        return mu, sd

    def predict(self, ratios):
        """Point estimate and 95% bounds, all in the ORIGINAL units.
        For log-scaled properties the bounds are asymmetric and always > 0."""
        mu, sd = self._predict_z(ratios)
        return self._bwd(mu), self._bwd(mu - Z * sd), self._bwd(mu + Z * sd)

    def _objective(self, ratios, target_z):
        mu, _ = self._predict_z(ratios)
        return float(np.sum(((mu[0] - target_z) / self.target_scales) ** 2))

    # ---- inverse design ----
    def suggest(self, targets, n_random_starts=16, seed=42):
        if isinstance(targets, dict):
            target = np.array([targets[c] for c in self.target_cols], dtype=float)
        else:
            target = np.asarray(targets, dtype=float)
        for i, c in enumerate(self.target_cols):
            if self._log_mask[i] and target[i] <= 0:
                raise ValueError(f"Target for '{c}' must be greater than 0.")
        target_z = self._fwd(target)

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
        # Starts: midpoint, random points, AND every known recipe.
        starts = [_repair((lo + hi) / 2.0, lo, hi)]
        for _ in range(n_random_starts):
            starts.append(_repair(rng.uniform(lo, hi), lo, hi))
        for row in self._X:
            starts.append(_repair(row, lo, hi))

        best = None
        for x0 in starts:
            res = minimize(self._objective, x0, args=(target_z,), method='SLSQP',
                           bounds=bounds, constraints=constraints)
            x = _repair(res.x, lo, hi)
            val = self._objective(x, target_z)
            if best is None or val < best[0]:
                best = (val, x)

        recipe = best[1]
        point, low, high = self.predict(recipe)
        return {
            'recipe':    dict(zip(self.input_cols, recipe)),
            'predicted': dict(zip(self.target_cols, point[0])),
            'lo':        dict(zip(self.target_cols, low[0])),
            'hi':        dict(zip(self.target_cols, high[0])),
            'data_std':  dict(zip(self.target_cols, self.data_std)),
            'log_scaled': list(self.log_targets),
            'error':     best[0],
        }

    def data_coverage(self):
        """For each material, its range in the data and whether it varies at all."""
        out = {}
        for i, m in enumerate(self.input_cols):
            col = self._X[:, i]
            out[m] = {'min': float(col.min()), 'max': float(col.max()),
                      'varies': bool(col.max() - col.min() > 1e-9)}
        return out

    def recommended_rows(self):
        """Rough heuristic: ~10 experiments per material that varies."""
        n_vary = max(1, sum(d['varies'] for d in self.data_coverage().values()))
        return n_vary, max(25, 10 * n_vary)

    # ---- honest accuracy check ----
    def cv_report(self):
        """Leave-one-out cross-validation per property: R2 and RMSE.
        R2 near 1 = good; near 0 = no better than guessing the average;
        negative = worse than the average. Scored in the modelling space, so a
        log-scaled property is judged on relative (not absolute) error."""
        n = self._X.shape[0]
        report = {}
        for i, col in enumerate(self.target_cols):
            preds = np.zeros(n)
            for j in range(n):
                mask = np.arange(n) != j
                gp = _fit_gp(self._X[mask], self._z[mask, i], restarts=3)
                preds[j] = gp.predict(self._X[j:j + 1])[0]
            actual = self._z[:, i]
            ss_res = np.sum((actual - preds) ** 2)
            ss_tot = np.sum((actual - actual.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            # RMSE reported in original units for interpretability
            if self._log_mask[i]:
                a_o, p_o = np.exp(actual), np.exp(preds)
            else:
                a_o, p_o = actual, preds
            rmse = float(np.sqrt(np.mean((a_o - p_o) ** 2)))
            report[col] = {'r2': float(r2), 'rmse': rmse,
                           'log_scaled': bool(self._log_mask[i])}
        return report


if __name__ == '__main__':
    model = MixtureModel()
    model.fit(pd.read_csv('data.csv'))
    print(f"Trained on {model.n_rows} rows.\n")
    print("Leave-one-out accuracy:")
    for col, d in model.cv_report().items():
        tag = " (log)" if d['log_scaled'] else ""
        print(f"  {col}{tag}: R2={d['r2']:.2f}, RMSE={d['rmse']:.1f}")

    out = model.suggest({'Tensile Strength (MPa)': 41,
                         'Elongation at Break (%)': 250,
                         "Young's Modulus (MPa)": 1103})
    print("\nSuggested recipe:")
    for mat, frac in out['recipe'].items():
        print(f"  {mat}: {frac:.2%}")
    print("\nPredicted:")
    for col in out['predicted']:
        print(f"  {col}: {out['predicted'][col]:.2f} "
              f"(95%: {out['lo'][col]:.2f} to {out['hi'][col]:.2f})")
