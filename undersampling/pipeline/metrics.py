"""AUC and the profit curve (highest point and average over the 5%-9% band).

Both only use the order of the clients, so the inflated probabilities of undersampled models
don't affect them. Ties are broken by row order (rows are sorted by client), so results are
deterministic.
"""
import numpy as np
from scipy.stats import rankdata


def auc(y_true, scores):
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = rankdata(scores)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def profit_curve(baja2, scores, hit, cost):
    """Cumulative profit contacting clients from highest to lowest score."""
    order = np.argsort(-np.asarray(scores), kind='stable')
    gains = np.where(np.asarray(baja2)[order] == 1, hit, -cost)
    return np.cumsum(gains)


def profit_metrics(baja2, scores, hit, cost, band):
    curve = profit_curve(baja2, scores, hit, cost)
    n = len(curve)
    peak_idx = int(np.argmax(curve))
    band_values = [curve[max(int(round(f * n)), 1) - 1] for f in band]
    return {
        'profit_peak': float(curve[peak_idx]),
        'profit_peak_n': peak_idx + 1,
        'profit_band': float(np.mean(band_values)),
    }


def evaluate(split, scores, cfg):
    return {
        'auc': auc(split.y, scores),
        'auc_baja2': auc(split.baja2, scores),
        **profit_metrics(split.baja2, scores, cfg.profit_hit, cfg.profit_cost, cfg.band),
    }
