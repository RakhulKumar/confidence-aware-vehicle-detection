"""Figures for the paper, one function per result file. Every function takes
the result dict written by the corresponding script and returns a matplotlib
Figure. Buckets with too few objects to be read are drawn hatched and
labelled with their n, never hidden.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_COLORS = {"confidence": "#6b7280", "mondrian_rank": "#2563eb", "occlusion_aware": "#dc2626",
           "standard_threshold": "#6b7280", "global_confidence": "#f59e0b",
           "mondrian_confidence": "#2563eb", "global_occlusion_aware": "#dc2626"}
_LABELS = {"confidence": "lower threshold everywhere (baseline)", "mondrian_rank": "occlusion-conditional conformal rank",
           "occlusion_aware": "duplicate- and context-aware score", "standard_threshold": "standard threshold",
           "global_confidence": "global conformal", "mondrian_confidence": "occlusion-conditional conformal",
           "global_occlusion_aware": "conformal on duplicate- and context-aware score"}


def _legend(ax, **kwargs):
    """Legend only when there is something to label (no warning on empty data)."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(**kwargs)


def _bucket_labels(rows):
    return [f"{r['lo']:.1f}-{r['hi']:.1f}\nn={r.get('n', r.get('n_objects', 0))}" for r in rows]


def _err(value, ci):
    if value is None or ci is None or ci[0] is None or ci[1] is None:
        return [0, 0]
    return [max(0.0, value - ci[0]), max(0.0, ci[1] - value)]


def plot_occlusion_histogram(report: dict, title: str = "Occlusion severity of annotated vehicles"):
    """report: {"<dataset>/<role>": {"buckets": [{"lo","hi","n"}...], "adequate": bool}}"""
    keys = list(report)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(keys) + 3), 4))
    n_b = len(next(iter(report.values()))["buckets"])
    width = 0.8 / max(len(keys), 1)
    for i, k in enumerate(keys):
        counts = [b["n"] for b in report[k]["buckets"]]
        ax.bar(np.arange(n_b) + i * width, np.maximum(counts, 0.5), width,
               label=f"{k}{'' if report[k].get('adequate', True) else '  (underpowered)'}",
               hatch=None if report[k].get("adequate", True) else "//")
    b0 = next(iter(report.values()))["buckets"]
    ax.set_xticks(np.arange(n_b) + 0.4 - width / 2)
    ax.set_xticklabels([f"{b['lo']:.1f}-{b['hi']:.1f}" for b in b0])
    ax.set_yscale("log"); ax.set_xlabel("occlusion severity"); ax.set_ylabel("objects (log scale)")
    ax.set_title(title); _legend(ax, fontsize=8); fig.tight_layout()
    return fig


def plot_rq1(r: dict, title: str = "RQ1: misses and uncertainty vs. occlusion"):
    rows = [b for b in r["buckets"] if b["n"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(rows))
    for ax, key, ylabel in ((axes[0], "miss_rate", "miss rate at standard threshold"),
                            (axes[1], "mean_uncertainty", "mean uncertainty (1 - object score)")):
        vals = [b[key] for b in rows]
        err = np.array([_err(b[key], b.get(f"{key}_ci")) for b in rows]).T
        bars = ax.bar(x, vals, yerr=err, capsize=3, color="#2563eb")
        for bar, b in zip(bars, rows):
            if not b["reliable"]:
                bar.set_hatch("//"); bar.set_alpha(0.4)
        ax.set_xticks(x); ax.set_xticklabels(_bucket_labels(rows), fontsize=8)
        ax.set_xlabel("occlusion severity"); ax.set_ylabel(ylabel); ax.set_ylim(0, 1)
    m = r.get("logistic_miss_model", {})
    sub = f"Spearman {r['spearman_occlusion_uncertainty']:.2f}, partial {r['partial_spearman_given_size_and_class']:.2f}"
    if m.get("fitted"):
        sub += f"; odds of a miss x{m['odds_ratio_per_0.1_occlusion']:.2f} per +0.1 occlusion"
    if not r["power"]["adequate"]:
        sub = "UNDERPOWERED: " + r["power"]["message"]
    fig.suptitle(f"{title}\n{sub}", fontsize=10); fig.tight_layout()
    return fig


def plot_rq2(r: dict, title: str = "RQ2: recovering missed vehicles at matched false-alarm cost"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, ceil_key, name in ((axes[0], "recovery", "recoverable_ceiling", "all misses"),
                                    (axes[1], "recovery_occluded", "recoverable_ceiling_occluded", "occluded misses")):
        for pol, entry in r.get("policies", {}).items():
            pts = entry["curve"]
            if pts:
                ax.step([0] + [p["extra_fp_per_image"] for p in pts], [0] + [p[key] for p in pts],
                        where="post", color=_COLORS.get(pol), label=_LABELS.get(pol, pol))
        if r.get(ceil_key) is not None:
            ax.axhline(r[ceil_key], ls=":", color="k", lw=1)
            ax.text(0, r[ceil_key], " ceiling: misses that have a candidate", va="bottom", fontsize=7)
        ax.set_xlabel("extra false alarms per image"); ax.set_ylabel(f"fraction recovered ({name})")
        ax.set_ylim(0, 1)
    _legend(axes[0], fontsize=8, loc="lower right")
    fig.suptitle(f"{title}\n{r.get('n_missed_at_standard', 0)} misses at the standard threshold, "
                 f"{r.get('n_missed_occluded', 0)} of them occluded", fontsize=10)
    fig.tight_layout()
    return fig


def plot_coverage(rep: dict, title: str = "Recall of the deployed output (accepted + flagged) per occlusion bucket"):
    names = [n for n in ("standard_threshold", "global_confidence", "mondrian_confidence",
                         "global_occlusion_aware") if n in rep]
    rows0 = rep[names[0]]["by_occlusion_bucket"]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(names)
    x = np.arange(len(rows0))
    for i, n in enumerate(names):
        rows = rep[n].get("system_by_occlusion_bucket", rep[n]["by_occlusion_bucket"])
        vals = [b["coverage"] or 0 for b in rows]
        err = np.array([_err(b["coverage"], b.get("coverage_ci")) for b in rows]).T
        bars = ax.bar(x + i * width, vals, width, yerr=err, capsize=2, color=_COLORS.get(n), label=_LABELS.get(n, n))
        for bar, b in zip(bars, rows):
            if not b["reliable"]:
                bar.set_hatch("//"); bar.set_alpha(0.4)
    target = next((rep[n]["target_coverage"] for n in names if "target_coverage" in rep[n]), None)
    if target is not None:
        ax.axhline(target, ls="--", color="k", lw=1); ax.text(len(rows0) - 0.5, target, " target", va="bottom", fontsize=8)
    ax.set_xticks(x + 0.4 - width / 2); ax.set_xticklabels(_bucket_labels(rows0), fontsize=8)
    ax.set_xlabel("annotated occlusion severity"); ax.set_ylabel("fraction of vehicles found"); ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10); _legend(ax, fontsize=8, loc="lower left"); fig.tight_layout()
    return fig


def plot_validity(curve: list[dict], title: str = "Promised vs. delivered recall (test role)"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    t = [row["target_coverage"] for row in curve]
    lo = min(t) - 0.05
    ax.plot([lo, 1], [lo, 1], "k--", lw=1, label="promised")
    for n in ("global_confidence", "mondrian_confidence", "global_occlusion_aware"):
        pts = [(row["target_coverage"], row[n]["coverage"], row[n]["all_groups_achievable"])
               for row in curve if n in row and row[n]["coverage"] is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=_COLORS[n], label=_LABELS[n])
            for a, c, ok in pts:
                ax.plot(a, c, "o" if ok else "x", color=_COLORS[n])
    ax.set_xlabel("target recall (1 - alpha)"); ax.set_ylabel("recall on held-out vehicles")
    ax.set_title(title + "\n(x = target not certifiable for some group)", fontsize=9)
    _legend(ax, fontsize=8); fig.tight_layout()
    return fig


def plot_rq3(r: dict, title: str = "RQ3: what prediction-only calibration does not see"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sw = r["threshold_sweep"]
    th = [s["threshold"] for s in sw]
    for key, label, color in (("dece", "D-ECE (prediction-only)", "#6b7280"), ("ma_ece", "MA-ECE (miss-aware)", "#dc2626"),
                              ("oce_max", "OCE (object-level Brier)", "#2563eb")):
        axes[0].plot(th, [s[key] for s in sw], color=color, label=label)
        pref = r["threshold_each_metric_prefers"].get(key)
        if pref:
            axes[0].plot(pref["threshold"], pref["value"], "v", color=color)
    axes[0].plot(th, [1 - (s["recall"] or 0) for s in sw], ":", color="k", label="miss rate")
    axes[0].set_xlabel("confidence threshold"); axes[0].set_ylabel("error"); _legend(axes[0], fontsize=8)
    axes[0].set_title("▼ = threshold each metric prefers", fontsize=9)
    rows = [b for b in r["gap_decomposition_by_occlusion"] if b["n_objects"] > 0]
    x = np.arange(len(rows))
    axes[1].bar(x - 0.2, [b["share_of_objects"] for b in rows], 0.4, color="#9ca3af", label="share of vehicles")
    axes[1].bar(x + 0.2, [b["share_of_gap"] or 0 for b in rows], 0.4, color="#dc2626",
                label="share of the D-ECE blind spot")
    axes[1].set_xticks(x); axes[1].set_xticklabels(_bucket_labels(rows), fontsize=8)
    axes[1].set_xlabel("occlusion severity"); _legend(axes[1], fontsize=8)
    p = r["at_standard_threshold"]
    fig.suptitle(f"{title}\nat threshold {p['threshold']:.2f}: D-ECE {p['dece']:.3f}, MA-ECE {p['ma_ece']:.3f}, "
                 f"miss mass {p['miss_mass']:.3f}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_reliability(bins: list[dict], title: str = "Reliability diagram (emitted detections)"):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    pts = [(b["confidence"], b["accuracy"]) for b in bins if b["accuracy"] is not None]
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#2563eb")
    ax.set_xlabel("confidence"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10); fig.tight_layout()
    return fig


def plot_rq5(r: dict, title: str = "RQ5: why vehicles are missed, by occlusion"):
    rows = [b for b in r.get("cause_shares_by_occlusion_bucket", []) if b["n_missed"] > 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    causes = ("suppressed", "merged", "unproposed", "confused", "weak_other")
    colors = ("#2563eb", "#60a5fa", "#1e3a8a", "#dc2626", "#9ca3af")
    bottom = np.zeros(len(rows))
    for c, col in zip(causes, colors):
        vals = np.array([b[c] or 0 for b in rows])
        ax.bar(np.arange(len(rows)), vals, bottom=bottom, color=col, label=c)
        bottom += vals
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([f"{b['lo']:.1f}-{b['hi']:.1f}\nmissed={b['n_missed']}" for b in rows], fontsize=8)
    ax.set_xlabel("occlusion severity"); ax.set_ylabel("share of misses"); _legend(ax, fontsize=8, ncol=3)
    ax.set_title(title, fontsize=10); fig.tight_layout()
    return fig


def plot_detector_comparison(points: dict[str, dict], title: str = "Detector reliability under each metric"):
    names = sorted(points)
    metrics = (("dece", "D-ECE"), ("ma_ece", "MA-ECE"), ("oce_max", "OCE"), ("miss_mass", "miss mass"))
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [points[n][m] for m, _ in metrics]
        err = np.array([_err(points[n][m], points[n].get(f"{m}_ci")) for m, _ in metrics]).T
        ax.bar(np.arange(len(metrics)) + i * width, vals, width, yerr=err, capsize=3, label=n)
    ax.set_xticks(np.arange(len(metrics)) + 0.4 - width / 2); ax.set_xticklabels([l for _, l in metrics])
    ax.set_ylabel("lower is better"); ax.set_title(title, fontsize=10); _legend(ax, fontsize=8); fig.tight_layout()
    return fig


def plot_synthetic_probe(r: dict, title: str = "Controlled occlusion probe (pasted occluders; NOT real data)"):
    rows = r["by_severity"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [b["severity"] for b in rows]
    for key, label, color in (("miss_rate", "miss rate at standard threshold", "#dc2626"),
                              ("mean_uncertainty", "mean uncertainty", "#2563eb")):
        y = [b[key] for b in rows]
        lo = [b[f"{key}_ci"][0] if b[f"{key}_ci"][0] is not None else b[key] for b in rows]
        hi = [b[f"{key}_ci"][1] if b[f"{key}_ci"][1] is not None else b[key] for b in rows]
        ax.plot(x, y, "o-", color=color, label=label); ax.fill_between(x, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel("constructed occlusion severity (same vehicles at every level)"); ax.set_ylim(0, 1)
    ax.set_title(f"{title}\n{r['n_targets']} target vehicles", fontsize=9); _legend(ax, fontsize=8); fig.tight_layout()
    return fig


def save(fig, path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
