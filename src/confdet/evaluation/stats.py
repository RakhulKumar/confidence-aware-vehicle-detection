"""Statistical building blocks shared by the RQ modules.

The sampling unit in this project is the IMAGE, not the object: vehicles in
one frame share lighting, congestion, camera and annotator, so their outcomes
are correlated. Treating objects as independent makes every interval too
narrow and every p-value too small. All uncertainty statements therefore come
from an image-level (cluster) bootstrap: images are resampled with
replacement, and every object or candidate of a drawn image comes with it.

Implemented with resampling WEIGHTS (how many times each image was drawn),
which turns each bootstrap replicate into a weighted sum instead of a data
copy.
"""
from __future__ import annotations

import numpy as np


def bootstrap_image_weights(n_images: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """(n_boot, n_images) array of multinomial resampling counts."""
    rng = np.random.default_rng(seed)
    if n_images <= 0:
        return np.zeros((n_boot, 0))
    return rng.multinomial(n_images, np.full(n_images, 1.0 / n_images), size=n_boot).astype(float)


def percentile_ci(samples, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval, ignoring non-finite replicates. (nan, nan) if
    fewer than 20 usable replicates.
    """
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 20:
        return float("nan"), float("nan")
    lo, hi = np.quantile(s, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = float(np.sum(weights))
    return float(np.sum(values * weights) / w) if w > 0 else float("nan")


def bootstrap_mean_ci(values: np.ndarray, img: np.ndarray, boot_w: np.ndarray,
                      level: float = 0.95) -> tuple[float, float]:
    """CI for the mean of per-object `values`, resampling images."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or boot_w.shape[1] == 0:
        return float("nan"), float("nan")
    n_images = boot_w.shape[1]
    num = np.bincount(img, weights=values, minlength=n_images)
    den = np.bincount(img, minlength=n_images).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        reps = (boot_w @ num) / (boot_w @ den)
    return percentile_ci(reps, level)


def wilson_interval(k: float, n: float, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (independent trials)."""
    if n <= 0:
        return float("nan"), float("nan")
    from scipy.stats import norm

    z = float(norm.ppf(1 - (1 - level) / 2))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def logistic_irls(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None,
                  l2: float = 1e-4, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """Weighted logistic regression by iteratively reweighted least squares.

    X must already contain an intercept column if one is wanted (column 0 is
    treated as the intercept and is not penalised). A small ridge penalty
    keeps the fit defined under (quasi-)separation. Returns the coefficients.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    scale = max(float(w.sum()), 1e-12)
    pen = np.full(p, l2 * scale)
    pen[0] = 0.0
    beta = np.zeros(p)

    def objective(b):
        eta = X @ b
        return float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)) + 0.5 * np.sum(pen * b * b))

    current = objective(beta)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        grad = X.T @ (w * (mu - y)) + pen * beta
        hess = (X * (w * mu * (1 - mu))[:, None]).T @ X + np.diag(pen + 1e-10 * scale)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        t = 1.0
        while t > 1e-6:                                   # backtracking: never accept a worse point
            candidate = beta - t * step
            value = objective(candidate)
            if value <= current + 1e-12:
                break
            t *= 0.5
        else:
            break
        converged = abs(current - value) <= tol * (abs(current) + tol)
        beta, current = candidate, value
        if converged:
            break
    return beta


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x: np.ndarray, y: np.ndarray, covariates: np.ndarray) -> float:
    """Spearman correlation of x and y after removing, by least squares on
    ranks, what each shares with `covariates` (continuous columns are ranked;
    pass indicator columns as 0/1).
    """
    from scipy.stats import rankdata

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    Z = np.asarray(covariates, dtype=float).reshape(len(x), -1)
    if len(x) < Z.shape[1] + 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    cols = [np.ones(len(x))]
    for j in range(Z.shape[1]):
        col = Z[:, j]
        if np.ptp(col) == 0:
            continue
        is_indicator = np.all((col == 0) | (col == 1))
        cols.append(col if is_indicator else rankdata(col))
    A = np.column_stack(cols)
    rx, ry = rankdata(x), rankdata(y)
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def resample_rows_by_image(img: np.ndarray, n_images: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one cluster-bootstrap replicate (for statistics that
    cannot be written with weights, e.g. rank correlations).
    """
    order = np.argsort(img, kind="stable")
    counts = np.bincount(img, minlength=n_images)
    starts = np.concatenate([[0], np.cumsum(counts)])
    drawn = rng.integers(0, n_images, size=n_images)
    parts = [order[starts[i]:starts[i + 1]] for i in drawn if counts[i] > 0]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
