"""Memory-frugal streaming aggregates over a backed h5ad.

The full Norman matrix (111k cells x 33.7k genes) plus scanpy normalization
exceeds this machine's 7 GB RAM, so milestone 1 never materializes the full
cell-level matrix. Everything is accumulated in row chunks from a backed
reader:

  - pseudobulk COUNTS per perturbation (sparse indicator accumulation)
  - disjoint split-half pseudobulk counts (for the ceiling)
  - control-cell co-expression statistics (sum-based Pearson ingredients)

Profiles are produced AFTER aggregation: pseudobulk counts are
library-size normalized to 1e4 and log1p-transformed (standard pseudobulk
practice), which is what eval_contract scores.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

CHUNK = 4000


def open_backed(path) -> ad.AnnData:
    return ad.read_h5ad(path, backed="r")


def normalize_profiles(counts: pd.DataFrame) -> pd.DataFrame:
    """Row-wise library-size normalization to 1e4 + log1p of count profiles."""
    totals = counts.sum(axis=1).replace(0, np.nan)
    prof = counts.div(totals, axis=0) * 1e4
    return np.log1p(prof.fillna(0.0))


def _chunk_slices(n: int, chunk: int = CHUNK):
    for start in range(0, n, chunk):
        yield slice(start, min(start + chunk, n))


def pseudobulk_counts_stream(adata_b, labels: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    """Sum counts per perturbation label. Returns (counts_df, cells_per_pert)."""
    uniq, inv = np.unique(labels, return_inverse=True)
    n_genes = adata_b.shape[1]
    sums = np.zeros((len(uniq), n_genes), dtype=np.float64)
    for sl in _chunk_slices(adata_b.shape[0]):
        X = adata_b[sl].X
        X = X if sp.issparse(X) else sp.csr_matrix(X)
        ind = sp.csr_matrix(
            (np.ones(sl.stop - sl.start), (inv[sl], np.arange(sl.stop - sl.start))),
            shape=(len(uniq), sl.stop - sl.start),
        )
        part = ind @ X
        sums += np.asarray(part.todense()) if sp.issparse(part) else np.asarray(part)
    counts = pd.DataFrame(sums, index=uniq, columns=[str(g) for g in adata_b.var_names])
    cells = pd.Series(np.bincount(inv), index=uniq, name="n_cells")
    return counts, cells


def split_half_counts_stream(
    adata_b, labels: np.ndarray, half: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pseudobulk counts for two disjoint cell halves.

    `half` is a per-cell array of 0/1 assignments (already restricted to the
    perturbations of interest; other cells may carry -1 and are ignored).
    """
    uniq = np.unique(labels[half >= 0])
    pos = {p: i for i, p in enumerate(uniq)}
    n_genes = adata_b.shape[1]
    sums = [np.zeros((len(uniq), n_genes)), np.zeros((len(uniq), n_genes))]
    for sl in _chunk_slices(adata_b.shape[0]):
        lab = labels[sl]
        h = half[sl]
        keep = h >= 0
        if not keep.any():
            continue
        rows = np.where(keep)[0]
        X = adata_b[sl].X
        X = X if sp.issparse(X) else sp.csr_matrix(X)
        X = X[rows]
        lab_k = np.array([pos[p] for p in lab[keep]])
        for side in (0, 1):
            sel = h[keep] == side
            if not sel.any():
                continue
            ind = sp.csr_matrix(
                (np.ones(sel.sum()), (lab_k[sel], np.arange(sel.sum()))),
                shape=(len(uniq), sel.sum()),
            )
            part = ind @ X[sel]
            sums[side] += (
                np.asarray(part.todense()) if sp.issparse(part) else np.asarray(part)
            )
    cols = [str(g) for g in adata_b.var_names]
    return (
        pd.DataFrame(sums[0], index=uniq, columns=cols),
        pd.DataFrame(sums[1], index=uniq, columns=cols),
    )


def control_corr_features_stream(
    adata_b, labels: np.ndarray, control_label: str, eval_genes: list[str]
) -> pd.DataFrame:
    """Pearson correlation of every gene with every eval gene, control cells only.

    Computed on log-normalized (1e4) control-cell expression via streamed
    sufficient statistics (sums, squared sums, and the gene x eval-gene gram).
    Returns DataFrame(index=all genes, columns=eval_genes).
    """
    var_names = [str(g) for g in adata_b.var_names]
    eval_idx = np.array([var_names.index(g) for g in eval_genes])
    n_genes = adata_b.shape[1]
    sum_x = np.zeros(n_genes)
    sum_x2 = np.zeros(n_genes)
    gram = np.zeros((n_genes, len(eval_genes)))
    n = 0
    ctrl_rows = np.where(labels == control_label)[0]
    for start in range(0, len(ctrl_rows), CHUNK):
        rows = ctrl_rows[start : start + CHUNK]
        X = adata_b[rows].X
        X = X if sp.issparse(X) else sp.csr_matrix(X)
        X = X.tocsr().astype(np.float64)
        rowsums = np.asarray(X.sum(axis=1)).ravel()
        rowsums[rowsums == 0] = 1.0
        X = X.multiply(1e4 / rowsums[:, None]).tocsr()
        X.data = np.log1p(X.data)
        sum_x += np.asarray(X.sum(axis=0)).ravel()
        sum_x2 += np.asarray(X.multiply(X).sum(axis=0)).ravel()
        part = X.T @ X[:, eval_idx]
        gram += np.asarray(part.todense()) if sp.issparse(part) else np.asarray(part)
        n += X.shape[0]
    cov = gram / n - np.outer(sum_x / n, sum_x[eval_idx] / n)
    var_all = sum_x2 / n - (sum_x / n) ** 2
    denom = np.sqrt(np.outer(var_all, var_all[eval_idx]))
    denom[denom == 0] = np.inf
    return pd.DataFrame(cov / denom, index=var_names, columns=eval_genes)
