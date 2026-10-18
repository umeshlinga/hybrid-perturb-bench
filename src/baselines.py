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
from sklearn.linear_model import Ridge


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


def predict_linear_ridge(
    real_pb: pd.DataFrame,
    train_perts: list[str],
    test_perts: list[str],
    feats: pd.DataFrame,
    control_label: str = "control",
    genes: list[str] | None = None,
    alpha: float = 100.0,
) -> pd.DataFrame:
    """Ridge on pseudobulk deltas with precomputed per-target-gene features.

    `feats` is a DataFrame(index=target gene, columns=eval genes) — for
    milestone 1, the control-cell co-expression profile of the target gene
    (see streaming.control_corr_features_stream). Callers guarantee every
    perturbation in train/test is present in feats.index.
    """
    genes = genes or list(real_pb.columns)
    ctrl = real_pb.loc[control_label, genes].to_numpy()

    X_train = feats.loc[train_perts, genes].to_numpy()
    Y_train = real_pb.loc[train_perts, genes].to_numpy() - ctrl
    model = Ridge(alpha=alpha)
    model.fit(X_train, Y_train)
    delta_test = model.predict(feats.loc[test_perts, genes].to_numpy())

    # Full-gene prediction: eval genes from the ridge, all other genes fall
    # back to the train mean delta so expression-space metrics on the full
    # gene set remain defined for every method.
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
