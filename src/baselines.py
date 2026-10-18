"""Three honest baselines for held-out perturbations (milestone 1).

All baselines return a predicted pseudobulk DataFrame (rows = test
perturbations + control, cols = genes) on the same log-normalized scale as
the real pseudobulk, so they plug straight into eval_contract.

  (a) control_mean     : predict the control profile for every perturbation
                         (the "no-change" baseline; predicted delta = 0).
  (b) global_mean      : predict the mean of the TRAIN perturbations'
                         profiles for every test perturbation.
  (c) linear_ridge     : ridge regression on pseudobulk deltas. Features for
                         a perturbation = co-expression profile of its target
                         gene in CONTROL cells only (correlation of the
                         target gene with every gene in the eval gene set).
                         Trained on train perturbations, predicts deltas for
                         unseen test perturbations. Alpha fixed a priori
                         (100.0), never tuned on test.

For CRISPRa (Norman 2019), the "target gene" of a single perturbation is the
perturbed gene itself. Combo perturbations are out of scope for milestone 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge


def _to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.todense(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def predict_control_mean(
    real_pb: pd.DataFrame, test_perts: list[str], control_label: str = "control"
) -> pd.DataFrame:
    ctrl = real_pb.loc[control_label]
    rows = {control_label: ctrl}
    for p in test_perts:
        rows[p] = ctrl
    return pd.DataFrame(rows).T.loc[:, real_pb.columns]


def predict_global_mean(
    real_pb: pd.DataFrame,
    train_perts: list[str],
    test_perts: list[str],
    control_label: str = "control",
) -> pd.DataFrame:
    gmean = real_pb.loc[train_perts].mean(axis=0)
    rows = {control_label: real_pb.loc[control_label]}
    for p in test_perts:
        rows[p] = gmean
    return pd.DataFrame(rows).T.loc[:, real_pb.columns]


def _control_coexpression_features(
    adata, pert_col: str, control_label: str, genes: list[str]
) -> pd.DataFrame:
    """Correlation of every gene with every eval gene, in control cells only.

    Returns a DataFrame indexed by gene name (potential target genes),
    columns = eval genes. Computed from control cells only, so no
    information from any perturbed cell (train or test) enters the features.
    Uses the sum-based Pearson formula on the sparse matrix, so the control
    expression matrix is never densified or centered in memory.
    """
    ctrl = adata[adata.obs[pert_col].astype(str) == control_label]
    X = ctrl.X if sp.issparse(ctrl.X) else sp.csr_matrix(ctrl.X)
    X = X.tocsr().astype(np.float64)
    var_names = [str(g) for g in ctrl.var_names]
    eval_idx = np.array([var_names.index(g) for g in genes])
    n = X.shape[0]
    sum_x = np.asarray(X.sum(axis=0)).ravel()              # (n_genes,)
    sum_x2 = np.asarray(X.multiply(X).sum(axis=0)).ravel()  # (n_genes,)
    gram = (X.T @ X[:, eval_idx])                          # (n_genes, n_eval)
    gram = np.asarray(gram.todense()) if sp.issparse(gram) else np.asarray(gram)
    cov = gram / n - np.outer(sum_x / n, sum_x[eval_idx] / n)
    var_all = sum_x2 / n - (sum_x / n) ** 2
    var_eval = var_all[eval_idx]
    denom = np.sqrt(np.outer(var_all, var_eval))
    denom[denom == 0] = np.inf
    corr = cov / denom
    return pd.DataFrame(corr, index=var_names, columns=genes)


def predict_linear_ridge(
    adata,
    real_pb: pd.DataFrame,
    train_perts: list[str],
    test_perts: list[str],
    pert_col: str = "perturbation",
    control_label: str = "control",
    genes: list[str] | None = None,
    alpha: float = 100.0,
    seed: int = 0,
) -> pd.DataFrame:
    genes = genes or list(real_pb.columns)
    feats = _control_coexpression_features(adata, pert_col, control_label, genes)
    ctrl = real_pb.loc[control_label, genes].to_numpy()

    def feats_for(perts):
        # single perturbations: target gene = the perturbation label itself
        return feats.loc[perts, genes].to_numpy()

    X_train = feats_for(train_perts)
    Y_train = real_pb.loc[train_perts, genes].to_numpy() - ctrl
    model = Ridge(alpha=alpha)
    model.fit(X_train, Y_train)
    delta_test = model.predict(feats_for(test_perts))

    # Full-gene prediction: HVG/eval genes from the ridge, all other genes
    # fall back to the train mean delta so expression-space metrics on the
    # full gene set remain defined for every method.
    train_delta_full = real_pb.loc[train_perts].to_numpy() - real_pb.loc[control_label].to_numpy()
    global_delta = train_delta_full.mean(axis=0)
    pred = pd.DataFrame(index=[control_label] + list(test_perts), columns=real_pb.columns, dtype=float)
    pred.loc[control_label] = real_pb.loc[control_label].to_numpy()
    gene_pos = {g: i for i, g in enumerate(real_pb.columns)}
    for row, pert in enumerate(test_perts):
        delta = global_delta.copy()
        for j, g in enumerate(genes):
            delta[gene_pos[g]] = delta_test[row, j]
        pred.loc[pert] = real_pb.loc[control_label].to_numpy() + delta
    return pred
