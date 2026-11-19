"""RQ4 -- does the calibration behaviour carry over to an independent
dataset with the detector frozen?   docs/theoretical_foundations.md section 7.4.

Two taxonomies are never compared by class index. Both the detector's classes
and the target dataset's classes are mapped to a common taxonomy
(configs/experiment.yaml -> evaluation.common_classes and each dataset's
class_map), and every transfer number is ALSO reported class-agnostically
("was a vehicle found here at all"), which is immune to taxonomy mismatch.

Three settings are compared on the transfer dataset's test role:

  frozen          detector AND post-hoc layer exactly as fitted on the
                  development dataset. Conformal guarantees assume
                  exchangeability and do not survive a dataset shift; this
                  row measures how badly they break.
  recalibrated    detector frozen; thresholds re-fitted on the transfer
                  dataset's calib role (no detector retraining -- the whole
                  point of a post-hoc layer). Only if that role exists.
  standard        the detector's plain thresholded output.

A gate comes first: if the detector proposes a candidate for fewer than half
the vehicles, the dataset is flagged and nothing else should be read.
"""
from __future__ import annotations


def _get(d: dict, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def coverage_rows(dev_conformal: dict | None, frozen: dict | None, recalibrated: dict | None,
                  tolerance: float = 0.05) -> dict:
    """One row per conformal rule: target, coverage on development / transfer
    frozen / transfer recalibrated, and next to EVERY coverage the flags per
    image it costs and whether the rule's target was certifiable (all context
    groups achievable). A coverage that equals the frozen one after
    recalibration usually means the target was not certifiable there; the
    certifiability column makes that visible instead of leaving it implied.
    """
    def side(rep):
        if not isinstance(rep, dict):
            return None, None, None
        return rep.get("coverage"), rep.get("flagged_per_image"), rep.get("all_groups_achievable")

    out = {}
    for rule, rep in (frozen or {}).items():
        if rule == "standard_threshold" or not isinstance(rep, dict):
            continue
        target = rep.get("target_coverage")
        dev_cov, dev_flags, dev_cert = side((dev_conformal or {}).get(rule))
        got, frozen_flags, frozen_cert = side(rep)
        recal, recal_flags, recal_cert = side((recalibrated or {}).get(rule))
        out[rule] = {
            "target": target,
            "development": dev_cov, "development_flags_per_image": dev_flags,
            "development_certifiable": dev_cert,
            "transfer_frozen": got, "transfer_frozen_flags_per_image": frozen_flags,
            "transfer_frozen_certifiable": frozen_cert,
            "transfer_recalibrated": recal, "transfer_recalibrated_flags_per_image": recal_flags,
            "transfer_recalibrated_certifiable": recal_cert,
            "frozen_guarantee_holds": (got is not None and target is not None and got >= target - tolerance),
            "recalibration_restores": (recal is not None and target is not None and recal >= target - tolerance)
            if recal is not None else None,
        }
    return out


def transfer_summary(dev: dict, transfer: dict, tolerance: float = 0.05) -> dict:
    """Side-by-side of two bundles produced by scripts/07 (development test
    role vs transfer test role). Pure bookkeeping.

    `conformal_coverage` is at the headline alpha. If both bundles carry
    `conformal_by_alpha` ({alpha: report} on development; {alpha: {"frozen":
    report, "recalibrated": report}} on transfer), the same rows are also
    given per alpha under `conformal_coverage_by_alpha`.
    """
    rows = {}
    for key, path in {
        "recall_at_standard": ("detection", "recall_at_standard"),
        "recall_at_floor": ("detection", "recall_at_floor"),
        "precision_at_standard": ("detection", "precision_at_standard"),
        "ap50": ("detection", "ap50"),
        "dece": ("calibration", "dece"),
        "ma_ece": ("calibration", "ma_ece"),
        "miss_mass": ("calibration", "miss_mass"),
        "oce_max": ("calibration", "oce_max"),
    }.items():
        a, b = _get(dev, *path), _get(transfer, *path)
        rows[key] = {"development": a, "transfer": b,
                     "change": (b - a) if a is not None and b is not None else None}

    coverage = coverage_rows(_get(dev, "conformal"), _get(transfer, "conformal_frozen"),
                             _get(transfer, "conformal_recalibrated"), tolerance)
    by_alpha = {}
    for a, reps in (_get(transfer, "conformal_by_alpha", default={}) or {}).items():
        by_alpha[a] = coverage_rows(_get(dev, "conformal_by_alpha", a), reps.get("frozen"),
                                    reps.get("recalibrated"), tolerance)
    out = {"metrics": rows, "conformal_coverage": coverage,
           "usable": _get(transfer, "gate", "usable"), "gate_message": _get(transfer, "gate", "message")}
    if by_alpha:
        out["conformal_coverage_by_alpha"] = by_alpha
    return out
