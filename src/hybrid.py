"""The hybrid model (milestone 2).

Prediction target: the pseudobulk expression DELTA (perturbation profile
minus control profile) on the eval gene set, for a perturbation that was
never seen in training.

Architecture (all choices fixed a priori, seed 0; every fitted statistic —
PCA bases, standardizers, the delta basis — is fit on the 84 TRAIN single
perturbations only):

Inputs per perturbed gene g (for a combo, the two component genes'
blocks are averaged before the MLP — composition by representation
averaging):

  1. esm block   : ESM-2 (esm2_t6_8M_UR50D) protein embedding of g,
                   standardized, PCA to 64 dims. Genes with no UniProt
                   sequence get the neutral (train-mean) vector.
  2. coex block  : g's co-expression profile in CONTROL cells only (the
                   `feats` frame from milestone 1), standardized,
                   PCA to 32 dims. Genes absent from the expression
                   matrix get the neutral vector.
  3. stats block : per-gene statistics from control + train data only —
                   control expression of g, variance of g's profile
                   across control+train perturbations, and summaries of
                   g's control-cell co-expression profile (mean |r|,
                   max |r|, fraction |r| > 0.3, g's co-expression with
                   itself when g is an eval gene).
  4. retrieval block ("pseudobulk delta" channel): similarity-weighted
                   mean of TRAIN perturbations' real pseudobulk deltas,
                   weights = softmax over cosine similarity (esm+coex
                   blocks) to train target genes, top-8, temperature
                   0.1; projected onto the train-delta PCA basis.

Output: coefficients in the train-delta PCA basis (40 components);
prediction = train mean delta + coefficients @ basis.

The MLP (PyTorch, CPU): in -> 128 -> ReLU -> Dropout(0.25) -> 128 ->
ReLU -> Dropout(0.25) -> 40. Adam (lr 1e-3, weight decay 1e-2),
full-batch MSE on standardized coefficients, 500 epochs, seed 0.

The retrieval channel alone (its weighted-mean delta, no MLP) is also
exported as the `retrieval_knn` reference predictor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

SEED = 0
K_DELTA = 40
ESM_PCA = 64
COEX_PCA = 32
HIDDEN = 128
DROPOUT = 0.25
EPOCHS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-2
RETRIEVAL_TOPK = 8
RETRIEVAL_TEMP = 0.1


def _pca_fit(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, components[k, d], explained) via SVD of centered X."""
    mean = X.mean(axis=0)
    Xc = X - mean
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, vt.shape[0])
    return mean, vt[:k], Xc


def _pca_apply(X: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    return (X - mean) @ comps.T


class HybridFeatures:
    """Builds per-perturbation input blocks; fitted on train singles only."""

    def __init__(
        self,
        real_pb: pd.DataFrame,
        feats: pd.DataFrame,
        eval_genes: list[str],
        train_perts: list[str],
        esm: dict[str, np.ndarray],
        control_label: str = "control",
    ):
        self.real_pb = real_pb
        self.feats = feats
        self.eval_genes = eval_genes
        self.train_perts = list(train_perts)
        self.esm = esm
        self.control_label = control_label

        ctrl = real_pb.loc[control_label]
        self.ctrl_eval = ctrl[eval_genes].to_numpy()
        deltas = real_pb.loc[train_perts, eval_genes].to_numpy() - self.ctrl_eval
        self.train_deltas = deltas
        # Delta basis (output space).
        self.delta_mean, self.delta_basis, _ = _pca_fit(deltas, K_DELTA)
        self.delta_coef_mean = np.zeros(self.delta_basis.shape[0])
        coef = _pca_apply(deltas, self.delta_mean, self.delta_basis)
        self.delta_coef_std = coef.std(axis=0) + 1e-8

        # ESM block basis.
        esm_dim = len(next(iter(esm.values()))) if esm else 0
        self.esm_dim = esm_dim
        train_esm = np.stack([self._esm_raw(g) for g in train_perts])
        self.esm_vec_mean = train_esm.mean(axis=0)
        self.esm_vec_std = train_esm.std(axis=0) + 1e-8
        train_esm_std = (train_esm - self.esm_vec_mean) / self.esm_vec_std
        self.esm_pca_mean, self.esm_pca_basis, _ = _pca_fit(train_esm_std, ESM_PCA)

        # Co-expression block basis.
        train_coex = np.stack([self._coex_raw(g) for g in train_perts])
        self.coex_vec_mean = train_coex.mean(axis=0)
        self.coex_vec_std = train_coex.std(axis=0) + 1e-8
        train_coex_std = (train_coex - self.coex_vec_mean) / self.coex_vec_std
        self.coex_pca_mean, self.coex_pca_basis, _ = _pca_fit(train_coex_std, COEX_PCA)

        # Stats block standardizer.
        train_stats = np.stack([self._stats_raw(g) for g in train_perts])
        self.stats_mean = train_stats.mean(axis=0)
        self.stats_std = train_stats.std(axis=0) + 1e-8

        # Retrieval geometry: standardized esm+coex PCA space.
        self._train_repr = np.stack([self._repr_single(g) for g in train_perts])
        self._train_repr_norm = self._train_repr / (
            np.linalg.norm(self._train_repr, axis=1, keepdims=True) + 1e-8
        )

    # ---- raw per-gene blocks -------------------------------------------------
    def _esm_raw(self, gene: str) -> np.ndarray:
        if gene in self.esm:
            return self.esm[gene].astype(np.float64)
        return np.zeros(self.esm_dim)

    def _coex_raw(self, gene: str) -> np.ndarray:
        if gene in self.feats.index:
            return self.feats.loc[gene, self.eval_genes].to_numpy(dtype=np.float64)
        return np.zeros(len(self.eval_genes))

    def _stats_raw(self, gene: str) -> np.ndarray:
        ctrl = self.real_pb.loc[self.control_label]
        if gene in self.real_pb.columns:
            ctrl_expr = float(ctrl[gene])
            prof_var = float(self.real_pb.loc[[self.control_label] + self.train_perts, gene].var())
        else:
            ctrl_expr = float(ctrl.mean())
            prof_var = float(self.real_pb.loc[[self.control_label] + self.train_perts].var().mean())
        if gene in self.feats.index:
            row = np.abs(self._coex_raw(gene))
            mean_abs = float(row.mean())
            max_abs = float(row.max())
            frac_strong = float((row > 0.3).mean())
            self_coex = (
                float(self.feats.loc[gene, gene]) if gene in self.eval_genes else 0.0
            )
        else:
            mean_abs = max_abs = frac_strong = self_coex = 0.0
        return np.array([ctrl_expr, prof_var, mean_abs, max_abs, frac_strong, self_coex])

    # ---- transformed blocks --------------------------------------------------
    def _esm_block(self, gene: str) -> np.ndarray:
        v = (self._esm_raw(gene) - self.esm_vec_mean) / self.esm_vec_std
        if gene not in self.esm:
            v = np.zeros_like(v)  # neutral: train mean embedding
        return _pca_apply(v[None, :], self.esm_pca_mean, self.esm_pca_basis)[0]

    def _coex_block(self, gene: str) -> np.ndarray:
        v = (self._coex_raw(gene) - self.coex_vec_mean) / self.coex_vec_std
        if gene not in self.feats.index:
            v = np.zeros_like(v)  # neutral: train mean profile
        return _pca_apply(v[None, :], self.coex_pca_mean, self.coex_pca_basis)[0]

    def _stats_block(self, gene: str) -> np.ndarray:
        return (self._stats_raw(gene) - self.stats_mean) / self.stats_std

    def _repr_single(self, gene: str) -> np.ndarray:
        return np.concatenate([self._esm_block(gene), self._coex_block(gene)])

    # ---- retrieval -----------------------------------------------------------
    def retrieval_delta(self, genes: list[str]) -> np.ndarray:
        """Similarity-weighted mean of train deltas for a 1- or 2-gene pert."""
        reprs = np.stack([self._repr_single(g) for g in genes]).mean(axis=0)
        reprs = reprs / (np.linalg.norm(reprs) + 1e-8)
        sims = self._train_repr_norm @ reprs
        top = np.argsort(-sims)[:RETRIEVAL_TOPK]
        w = np.exp(sims[top] / RETRIEVAL_TEMP)
        w = w / w.sum()
        return w @ self.train_deltas[top]

    def input_vector(self, genes: list[str]) -> np.ndarray:
        esm_b = np.stack([self._esm_block(g) for g in genes]).mean(axis=0)
        coex_b = np.stack([self._coex_block(g) for g in genes]).mean(axis=0)
        stats_b = np.stack([self._stats_block(g) for g in genes]).mean(axis=0)
        retr = self.retrieval_delta(genes)
        retr_coef = _pca_apply(
            retr[None, :], self.delta_mean, self.delta_basis
        )[0] / self.delta_coef_std
        return np.concatenate([esm_b, coex_b, stats_b, retr_coef])

    def target_coef(self, pert: str) -> np.ndarray:
        delta = self.real_pb.loc[pert, self.eval_genes].to_numpy() - self.ctrl_eval
        coef = _pca_apply(delta[None, :], self.delta_mean, self.delta_basis)[0]
        return coef / self.delta_coef_std

    def coef_to_delta(self, coef_std: np.ndarray) -> np.ndarray:
        coef = coef_std * self.delta_coef_std
        return self.delta_mean + coef @ self.delta_basis


class HybridMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_hybrid(features: HybridFeatures) -> HybridMLP:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X = np.stack([features.input_vector([g]) for g in features.train_perts])
    Y = np.stack([features.target_coef(g) for g in features.train_perts])
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)
    model = HybridMLP(x.shape[1], y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch + 1}/{EPOCHS} train MSE (std coef) {loss.item():.4f}", flush=True)
    model.eval()
    return model


def predict_delta(features: HybridFeatures, model: HybridMLP, genes: list[str]) -> np.ndarray:
    """Predicted delta on the eval gene set for a 1- or 2-gene perturbation."""
    v = torch.tensor(features.input_vector(genes)[None, :], dtype=torch.float32)
    with torch.no_grad():
        coef = model(v).numpy()[0]
    return features.coef_to_delta(coef)


def to_full_frame(
    real_pb: pd.DataFrame,
    eval_genes: list[str],
    deltas_by_pert: dict[str, np.ndarray],
    train_perts: list[str],
    control_label: str = "control",
) -> pd.DataFrame:
    """Assemble a full-gene prediction frame: eval genes from the model,
    all other genes fall back to the train mean delta (same convention as
    the milestone-1 ridge), so all-gene metrics are defined."""
    ctrl = real_pb.loc[control_label]
    global_delta = (real_pb.loc[train_perts].to_numpy() - ctrl.to_numpy()).mean(axis=0)
    gene_pos = {g: i for i, g in enumerate(real_pb.columns)}
    pred = pd.DataFrame(
        index=[control_label] + list(deltas_by_pert), columns=real_pb.columns, dtype=float
    )
    pred.loc[control_label] = ctrl.to_numpy()
    for pert, delta in deltas_by_pert.items():
        full = global_delta.copy()
        for j, g in enumerate(eval_genes):
            full[gene_pos[g]] = delta[j]
        pred.loc[pert] = ctrl.to_numpy() + full
    return pred
