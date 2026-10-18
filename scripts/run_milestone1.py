"""Milestone 1 runner: Norman 2019 smoke test — eval contract + 3 baselines.

Pipeline (all choices fixed a priori, seed 0):
  1. Load NormanWeissman2019_filtered.h5ad (scPerturb / Zenodo 10044268).
  2. Detect perturbation column + control label; detect counts vs
     log-normalized X (normalize_total 1e4 + log1p only if counts).
  3. Keep control + SINGLE perturbations with >= 30 cells (combos deferred).
  4. Headline split: hold out 20% of single perturbations entirely (seed 0).
  5. Eval gene set: top-2000 HVGs from control+train cells only (headline);
     all-gene metrics are also written as secondary rows.
  6. Run baselines (a) control mean, (b) global perturbation mean,
     (c) linear ridge on control co-expression features; score with
     src/eval_contract.py; compute the split-half ceiling on test perts.
  7. Write results/*.csv, results/split.json, predicted pseudobulk h5ads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import predict_control_mean, predict_global_mean, predict_linear_ridge
from eval_contract import aggregate, ceiling_split_half, per_perturbation_metrics, pseudobulk

DATA = ROOT / "data" / "NormanWeissman2019_filtered.h5ad"
RESULTS = ROOT / "results"
SEED = 0
TEST_FRACTION = 0.20
MIN_CELLS = 30
N_HVG = 2000
DEG_K = 50


def detect_pert_column(adata) -> tuple[str, str]:
    for cand in ["perturbation", "condition", "perturbation_name", "gene"]:
        if cand in adata.obs.columns:
            vals = adata.obs[cand].astype(str)
            if (vals.str.lower() == "control").any():
                return cand, vals[vals.str.lower() == "control"].iloc[0]
    for col in adata.obs.columns:
        vals = adata.obs[col].astype(str)
        if (vals.str.lower() == "control").any() and vals.nunique() < 1000:
            return col, vals[vals.str.lower() == "control"].iloc[0]
    raise RuntimeError(f"No perturbation column with a control label. obs cols: {list(adata.obs.columns)}")


def looks_like_counts(adata) -> bool:
    X = adata.X[:2000]
    vals = np.asarray(X.todense()).ravel() if sp.issparse(X) else np.asarray(X).ravel()
    nz = vals[vals > 0]
    if nz.size == 0:
        return False
    frac_int = np.mean(np.isclose(nz, np.round(nz)))
    return bool(frac_int > 0.99 and vals.max() > 30)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print(f"Loading {DATA} ...", flush=True)
    adata = ad.read_h5ad(DATA)
    print(f"shape: {adata.shape}")
    print(f"obs columns: {list(adata.obs.columns)}")

    pert_col, control_label = detect_pert_column(adata)
    print(f"perturbation column: {pert_col!r}; control label: {control_label!r}")

    if looks_like_counts(adata):
        print("X looks like raw counts -> normalize_total(1e4) + log1p")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("X looks log-normalized already -> using as-is")

    labels = adata.obs[pert_col].astype(str)
    counts = labels.value_counts()
    singles = [
        p
        for p in counts.index
        if p != control_label and "+" not in p and counts[p] >= MIN_CELLS
    ]
    combos = [p for p in counts.index if "+" in p]
    print(f"control cells: {counts[control_label]}")
    print(f"single perturbations with >= {MIN_CELLS} cells: {len(singles)}")
    print(f"combo perturbations (deferred this milestone): {len(combos)}")

    rng = np.random.default_rng(SEED)
    singles_sorted = sorted(singles)
    perm = rng.permutation(len(singles_sorted))
    n_test = max(10, int(round(TEST_FRACTION * len(singles_sorted))))
    test_perts = sorted(singles_sorted[i] for i in perm[:n_test])
    train_perts = sorted(singles_sorted[i] for i in perm[n_test:])
    print(f"train perturbations: {len(train_perts)}; test perturbations: {len(test_perts)}")

    keep = labels.isin([control_label] + singles_sorted)
    adata = adata[keep].copy()

    # HVGs from control + train cells ONLY (no test-perturbation information).
    train_mask = adata.obs[pert_col].astype(str).isin([control_label] + train_perts)
    sub = adata[train_mask].copy()
    sc.pp.highly_variable_genes(sub, n_top_genes=N_HVG, flavor="seurat", subset=False)
    hvg = sorted(sub.var_names[sub.var["highly_variable"]].astype(str))
    print(f"HVG eval gene set: {len(hvg)} genes")

    real_pb = pseudobulk(adata, pert_col)

    # Restrict ridge features/targets to perturbations whose target gene exists.
    var_set = set(adata.var_names.astype(str))
    ridge_train = [p for p in train_perts if p in var_set]
    ridge_test = [p for p in test_perts if p in var_set]
    if len(ridge_test) < len(test_perts) or len(ridge_train) < len(train_perts):
        print(
            f"ridge: dropped {len(train_perts) - len(ridge_train)} train / "
            f"{len(test_perts) - len(ridge_test)} test perts with target gene absent"
        )

    preds = {
        "control_mean": predict_control_mean(real_pb, test_perts, control_label),
        "global_mean": predict_global_mean(real_pb, train_perts, test_perts, control_label),
        "linear_ridge": predict_linear_ridge(
            adata, real_pb, ridge_train, ridge_test,
            pert_col=pert_col, control_label=control_label, genes=hvg,
        ),
    }

    # Save predicted + real pseudobulk AnnData (the contract's artifacts).
    def to_anndata(pb: pd.DataFrame, perts: list[str]) -> ad.AnnData:
        sub = pb.loc[[control_label] + perts]
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
        for gene_set_name, gene_list in [("hvg2000", hvg), ("all_genes", None)]:
            m = per_perturbation_metrics(pb, real_pb, control_label, DEG_K, gene_list)
            m = m.reset_index(names="perturbation")
            m.insert(0, "gene_set", gene_set_name)
            m.insert(0, "method", name)
            per_rows.append(m)
            a = aggregate(m.drop(columns=["method", "gene_set", "perturbation"]))
            a = a.reset_index(names="metric")
            a.insert(0, "gene_set", gene_set_name)
            a.insert(0, "method", name)
            agg_rows.append(a)

    # Ceiling on test perturbations (control + test cells only), HVG gene set.
    test_mask = adata.obs[pert_col].astype(str).isin([control_label] + test_perts)
    ceil_per, ceil_agg = ceiling_split_half(
        adata[test_mask], pert_col, control_label, DEG_K, hvg, seed=SEED
    )
    ceil_per = ceil_per.reset_index(names="perturbation")
    ceil_per.insert(0, "gene_set", "hvg2000")
    ceil_per.insert(0, "method", "ceiling_split_half")
    per_rows.append(ceil_per)
    ceil_agg = ceil_agg.reset_index(names="metric")
    ceil_agg.insert(0, "gene_set", "hvg2000")
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
        "n_hvg": len(hvg),
        "control_label": control_label,
        "pert_column": pert_col,
        "train_perturbations": train_perts,
        "test_perturbations": test_perts,
        "n_combos_deferred": len(combos),
    }
    (RESULTS / "split.json").write_text(json.dumps(split, indent=2))

    pd.set_option("display.width", 200)
    print("\n=== AGGREGATE (hvg2000, headline) ===")
    print(
        agg_df[agg_df["gene_set"] == "hvg2000"]
        .pivot_table(index="method", columns="metric", values="mean")
        .round(4)
        .to_string()
    )
    print("\n=== AGGREGATE (all_genes) ===")
    print(
        agg_df[agg_df["gene_set"] == "all_genes"]
        .pivot_table(index="method", columns="metric", values="mean")
        .round(4)
        .to_string()
    )
    print("\nWrote results to", RESULTS)


if __name__ == "__main__":
    main()
