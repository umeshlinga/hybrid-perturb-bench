"""Milestone 1 runner: Norman 2019 smoke test — eval contract + 3 baselines.

Memory-frugal design (the full 111k x 33.7k matrix + scanpy normalization
OOMs this 7 GB machine): the h5ad is only ever read BACKED, in row chunks.
Pseudobulk count profiles are accumulated by streaming; profiles are then
library-normalized (1e4) + log1p AFTER aggregation (standard pseudobulk
practice). See src/streaming.py.

Pipeline (all choices fixed a priori, seed 0):
  1. NormanWeissman2019_filtered.h5ad (scPerturb / Zenodo 10044268; GEO
     GSE133344), K562 CRISPRa, obs['perturbation'], control label 'control'.
  2. Keep control + SINGLE perturbations (nperts == 1) with >= 30 cells.
     Combos (nperts == 2) are deferred this milestone.
  3. Headline split: hold out 20% of single perturbations entirely (seed 0).
  4. Eval gene set: top-2000 genes by variance of pseudobulk profiles across
     control + TRAIN perturbations only (headline); all-gene metrics are
     also written as secondary rows.
  5. Baselines: (a) control mean, (b) global perturbation mean,
     (c) linear ridge — features = control-cell co-expression profile of
     the target gene (streamed, control cells only), alpha = 100 fixed.
  6. Ceiling: disjoint split-half of test-perturbation + control cells,
     Spearman-Brown corrected Pearson (mirrors Arc cell-eval --ceiling).
  7. Writes results/*.csv, results/split.json, pseudobulk h5ad artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import predict_control_mean, predict_global_mean, predict_linear_ridge
from eval_contract import aggregate, ceiling_from_pseudobulks, per_perturbation_metrics
from streaming import (
    control_corr_features_stream,
    normalize_profiles,
    open_backed,
    pseudobulk_counts_stream,
    split_half_counts_stream,
)

DATA = ROOT / "data" / "NormanWeissman2019_filtered.h5ad"
RESULTS = ROOT / "results"
SEED = 0
TEST_FRACTION = 0.20
MIN_CELLS = 30
N_EVAL_GENES = 2000
DEG_K = 50
PERT_COL = "perturbation"
CONTROL = "control"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print(f"Opening backed: {DATA}", flush=True)
    adata_b = open_backed(DATA)
    print(f"shape: {adata_b.shape}", flush=True)
    labels = adata_b.obs[PERT_COL].astype(str).to_numpy()
    nperts = adata_b.obs["nperts"].to_numpy()
    var_names = [str(g) for g in adata_b.var_names]

    counts_all = pd.Series(labels).value_counts()
    singles = sorted(
        p
        for p in counts_all.index
        if p != CONTROL
        and counts_all[p] >= MIN_CELLS
        and int(pd.Series(nperts[labels == p]).iloc[0]) == 1
    )
    n_combos = int(sum(1 for p in counts_all.index if p != CONTROL and p not in singles))
    print(f"control cells: {counts_all[CONTROL]}", flush=True)
    print(f"single perturbations (>= {MIN_CELLS} cells): {len(singles)}", flush=True)
    print(f"other perturbations deferred (combos / low-cell): {n_combos}", flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(singles))
    n_test = max(10, int(round(TEST_FRACTION * len(singles))))
    test_perts = sorted(singles[i] for i in perm[:n_test])
    train_perts = sorted(singles[i] for i in perm[n_test:])
    print(f"train: {len(train_perts)}; test: {len(test_perts)}", flush=True)

    print("Streaming pseudobulk counts ...", flush=True)
    counts_pb, cells_pb = pseudobulk_counts_stream(adata_b, labels)
    real_pb = normalize_profiles(counts_pb)
    print(f"pseudobulk profiles: {real_pb.shape}", flush=True)

    # Eval gene set: top-N by profile variance across control + train only.
    var_profile = real_pb.loc[[CONTROL] + train_perts].var(axis=0)
    eval_genes = sorted(var_profile.sort_values(ascending=False).head(N_EVAL_GENES).index)
    print(f"eval gene set: {len(eval_genes)} genes", flush=True)

    # Ridge feasibility: target gene must be measured.
    var_set = set(var_names)
    ridge_train = [p for p in train_perts if p in var_set]
    ridge_test = [p for p in test_perts if p in var_set]
    print(f"ridge coverage: train {len(ridge_train)}/{len(train_perts)}, test {len(ridge_test)}/{len(test_perts)}", flush=True)

    print("Streaming control co-expression features ...", flush=True)
    feats = control_corr_features_stream(adata_b, labels, CONTROL, eval_genes)

    preds = {
        "control_mean": predict_control_mean(real_pb, test_perts, CONTROL),
        "global_mean": predict_global_mean(real_pb, train_perts, test_perts, CONTROL),
        "linear_ridge": predict_linear_ridge(
            real_pb, ridge_train, ridge_test, feats, CONTROL, genes=eval_genes
        ),
    }

    def to_anndata(pb: pd.DataFrame, perts: list[str]) -> ad.AnnData:
        sub = pb.loc[[CONTROL] + perts]
        out = ad.AnnData(X=sub.to_numpy(dtype=np.float32))
        out.obs_names = sub.index.astype(str)
        out.var_names = sub.columns.astype(str)
        out.obs["perturbation"] = sub.index.astype(str)
        return out

    to_anndata(real_pb, test_perts).write_h5ad(RESULTS / "real_pseudobulk_test.h5ad")
    for name, pb in preds.items():
        perts = ridge_test if name == "linear_ridge" else test_perts
        to_anndata(pb, perts).write_h5ad(RESULTS / f"predictions_{name}.h5ad")

    per_rows, agg_rows = [], []
    for name, pb in preds.items():
        for gene_set_name, gene_list in [("eval2000", eval_genes), ("all_genes", None)]:
            m = per_perturbation_metrics(pb, real_pb, CONTROL, DEG_K, gene_list)
            m = m.reset_index(names="perturbation")
            m.insert(0, "gene_set", gene_set_name)
            m.insert(0, "method", name)
            per_rows.append(m)
            a = aggregate(m.drop(columns=["method", "gene_set", "perturbation"]))
            a = a.reset_index(names="metric")
            a.insert(0, "gene_set", gene_set_name)
            a.insert(0, "method", name)
            agg_rows.append(a)

    print("Streaming split-half ceiling ...", flush=True)
    rng = np.random.default_rng(SEED)
    half = np.full(len(labels), -1, dtype=int)
    for pert in [CONTROL] + test_perts:
        idx = np.where(labels == pert)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        half[idx[:h]] = 0
        half[idx[h : 2 * h]] = 1
    counts_a, counts_b = split_half_counts_stream(adata_b, labels, half)
    pb_a = normalize_profiles(counts_a)
    pb_b = normalize_profiles(counts_b)
    ceil_per, ceil_agg = ceiling_from_pseudobulks(pb_a, pb_b, CONTROL, DEG_K, eval_genes)
    ceil_per = ceil_per.reset_index(names="perturbation")
    ceil_per.insert(0, "gene_set", "eval2000")
    ceil_per.insert(0, "method", "ceiling_split_half")
    per_rows.append(ceil_per)
    ceil_agg = ceil_agg.reset_index(names="metric")
    ceil_agg.insert(0, "gene_set", "eval2000")
    ceil_agg.insert(0, "method", "ceiling_split_half")
    agg_rows.append(ceil_agg)

    per_df = pd.concat(per_rows, ignore_index=True)
    agg_df = pd.concat(agg_rows, ignore_index=True)
    per_df.to_csv(RESULTS / "milestone1_metrics_per_perturbation.csv", index=False)
    agg_df.to_csv(RESULTS / "milestone1_metrics_aggregate.csv", index=False)

    split = {
        "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "min_cells": MIN_CELLS,
        "deg_k": DEG_K,
        "n_eval_genes": len(eval_genes),
        "eval_gene_selection": "top-2000 by pseudobulk profile variance across control+train perturbations",
        "control_label": CONTROL,
        "pert_column": PERT_COL,
        "train_perturbations": train_perts,
        "test_perturbations": test_perts,
        "ridge_train_perturbations": ridge_train,
        "ridge_test_perturbations": ridge_test,
        "n_other_perturbations_deferred": n_combos,
        "cells_per_perturbation": {p: int(cells_pb[p]) for p in [CONTROL] + singles},
    }
    (RESULTS / "split.json").write_text(json.dumps(split, indent=2))

    pd.set_option("display.width", 250)
    for gs in ["eval2000", "all_genes"]:
        print(f"\n=== AGGREGATE means ({gs}) ===", flush=True)
        print(
            agg_df[agg_df["gene_set"] == gs]
            .pivot_table(index="method", columns="metric", values="mean")
            .round(4)
            .to_string()
        )
    print("\nDONE. Results in", RESULTS, flush=True)


if __name__ == "__main__":
    main()
