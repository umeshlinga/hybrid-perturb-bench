"""Evaluation contract for hybrid-perturb-bench (milestone 1).

Contract: given a PREDICTED AnnData and a REAL AnnData on the same
log-normalized scale, with the same genes and a perturbation label column,
score per-perturbation and aggregate agreement.

Milestone-1 scope: every baseline predicts a per-perturbation mean profile,
so the contract scores PSEUDOBULK profiles (per-perturbation means). Metrics
mirror the delta-expression family used by Arc cell-eval:

  - mse_delta / rmse_delta : MSE of (pred - control) vs (real - control)
  - pearson_delta          : Pearson r between predicted and real delta
  - mse_expr / pearson_expr: same, on raw profiles (not deltas)
  - deg_overlap_at_k       : |top-k |delta| genes(pred) ∩ top-k |delta|(real)| / k

The ceiling mirrors cell-eval's --ceiling idea locally: each perturbation's
cells (and the control's) are split into two DISJOINT halves; one half plays
"real", the other "prediction". The aggregate Pearson is Spearman-Brown
corrected (r' = 2r / (1 + r)) to map half-depth back to full depth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def pseudobulk(adata, pert_col: str) -> pd.DataFrame:
    """Mean expression per perturbation label. Rows = perturbations, cols = genes.

    Computed with a sparse indicator matmul (indicator^T @ X / counts) so the
    full cell x gene matrix is never densified.
    """
    labels = adata.obs[pert_col].astype(str).to_numpy()
    uniq, inv = np.unique(labels, return_inverse=True)
    n_cells, n_genes = adata.shape
    ind = sp.csr_matrix(
        (np.ones(n_cells), (inv, np.arange(n_cells))), shape=(len(uniq), n_cells)
    )
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    sums = ind @ X  # (n_perts, n_genes), sparse or dense depending on X
    sums = np.asarray(sums.todense()) if sp.issparse(sums) else np.asarray(sums)
    counts = np.asarray(ind.sum(axis=1)).ravel()
    pb = sums / counts[:, None]
    return pd.DataFrame(pb, index=uniq, columns=[str(g) for g in adata.var_names])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _topk_abs(v: np.ndarray, k: int) -> np.ndarray:
    k = min(k, v.size)
    return np.argpartition(-np.abs(v), k - 1)[:k]


def per_perturbation_metrics(
    pred_pb: pd.DataFrame,
    real_pb: pd.DataFrame,
    control_label: str = "control",
    deg_k: int = 50,
    genes: list[str] | None = None,
) -> pd.DataFrame:
    """Score predicted pseudobulk rows against real pseudobulk rows.

    Both frames must contain `control_label` and share gene columns.
    Metrics are computed on `genes` (default: all shared genes).
    """
    genes = genes or [g for g in real_pb.columns if g in pred_pb.columns]
    pred_pb = pred_pb.loc[:, genes]
    real_pb = real_pb.loc[:, genes]
    ctrl_pred = pred_pb.loc[control_label].to_numpy()
    ctrl_real = real_pb.loc[control_label].to_numpy()

    rows = {}
    for pert in real_pb.index:
        if pert == control_label or pert not in pred_pb.index:
            continue
        rp = real_pb.loc[pert].to_numpy()
        pp = pred_pb.loc[pert].to_numpy()
        rd = rp - ctrl_real
        pd_ = pp - ctrl_pred
        mse_delta = float(np.mean((pd_ - rd) ** 2))
        top_r = set(_topk_abs(rd, deg_k).tolist())
        top_p = set(_topk_abs(pd_, deg_k).tolist())
        rows[pert] = {
            "mse_delta": mse_delta,
            "rmse_delta": float(np.sqrt(mse_delta)),
            "pearson_delta": _pearson(pd_, rd),
            "mse_expr": float(np.mean((pp - rp) ** 2)),
            "pearson_expr": _pearson(pp, rp),
            "deg_overlap_at_k": len(top_r & top_p) / deg_k,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and median across perturbations for every metric column."""
    out = pd.DataFrame(
        {
            "mean": metrics.mean(numeric_only=True),
            "median": metrics.median(numeric_only=True),
        }
    )
    out["n_perturbations"] = len(metrics)
    return out


def spearman_brown(r: float) -> float:
    return float(2 * r / (1 + r)) if pd.notna(r) else np.nan


def ceiling_split_half(
    adata,
    pert_col: str,
    control_label: str = "control",
    deg_k: int = 50,
    genes: list[str] | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real-data ceiling: disjoint split-half self-evaluation.

    For every perturbation (and the control), cells are split into two
    disjoint halves. Pseudobulk of half A = 'real', half B = 'prediction'.
    Returns (per_perturbation_df, aggregate_df) where the aggregate Pearson
    columns are Spearman-Brown corrected.
    """
    rng = np.random.default_rng(seed)
    labels = adata.obs[pert_col].astype(str).to_numpy()
    idx_a, idx_b = [], []
    for pert in np.unique(labels):
        idx = np.where(labels == pert)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        idx_a.extend(idx[:half])
        idx_b.extend(idx[half : 2 * half])
    pb_a = pseudobulk(adata[np.sort(idx_a)], pert_col)
    pb_b = pseudobulk(adata[np.sort(idx_b)], pert_col)
    per_pert = per_perturbation_metrics(pb_b, pb_a, control_label, deg_k, genes)
    agg = aggregate(per_pert)
    for col in ["pearson_delta", "pearson_expr"]:
        if col in agg.index:
            agg.loc[col, "mean"] = spearman_brown(agg.loc[col, "mean"])
            agg.loc[col, "median"] = spearman_brown(agg.loc[col, "median"])
    agg["spearman_brown_corrected"] = agg.index.isin(["pearson_delta", "pearson_expr"])
    return per_pert, agg
