"""Bounded Arc cell-eval cross-check of the local contract (milestone 1).

Scope (deliberately small so it fits this 7 GB / 2-core box): control cells
(subsampled to 3000) + the first 5 test perturbations, restricted to the
2000-gene eval set. The prediction scored is the global_mean baseline at
cell level: every predicted cell of a perturbation gets that perturbation's
predicted pseudobulk profile. Outputs land in results/cell_eval_check/.
The point is agreement of the shared metric family (delta Pearson / MSE)
between Arc cell-eval and src/eval_contract.py — not a second leaderboard.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streaming import open_backed

DATA = ROOT / "data" / "NormanWeissman2019_filtered.h5ad"
RESULTS = ROOT / "results"
OUT = RESULTS / "cell_eval_check"
CONTROL = "control"
N_CONTROL = 3000
N_PERTS = 5
SEED = 0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = pickle.load(open(RESULTS / "_m1_cache.pkl", "rb"))
    real_pb, eval_genes = cache["real_pb"], cache["eval_genes"]
    import json

    split = json.loads((RESULTS / "split.json").read_text())
    test_perts = split["test_perturbations"][:N_PERTS]
    train_perts = split["train_perturbations"]
    global_profile = real_pb.loc[train_perts].mean(axis=0)

    adata_b = open_backed(DATA)
    labels = adata_b.obs["perturbation"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED)
    ctrl_idx = np.where(labels == CONTROL)[0]
    ctrl_idx = np.sort(rng.choice(ctrl_idx, size=min(N_CONTROL, len(ctrl_idx)), replace=False))
    pert_idx = np.where(np.isin(labels, test_perts))[0]
    rows = np.sort(np.concatenate([ctrl_idx, pert_idx]))
    print(f"cells: {len(rows)} (control {len(ctrl_idx)}, perts {len(pert_idx)})", flush=True)

    gene_idx = np.array([list(adata_b.var_names).index(g) for g in eval_genes])
    sub = adata_b[rows][:, gene_idx].to_memory()
    sub.var_names = eval_genes
    sub.obs["perturbation"] = labels[rows]
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    print(f"real subset: {sub.shape}", flush=True)

    pred = sub.copy()
    X = np.asarray(sub.X.todense()) if hasattr(sub.X, "todense") else np.asarray(sub.X)
    pred_profiles = {CONTROL: real_pb.loc[CONTROL, eval_genes].to_numpy()}
    for p in test_perts:
        pred_profiles[p] = global_profile[eval_genes].to_numpy()
    for i, lab in enumerate(sub.obs["perturbation"].astype(str)):
        X[i, :] = pred_profiles[lab]
    pred.X = X
    sub.write_h5ad(OUT / "real_subset.h5ad")
    pred.write_h5ad(OUT / "pred_global_mean_subset.h5ad")

    from cell_eval import MetricsEvaluator

    ev = MetricsEvaluator(
        adata_pred=str(OUT / "pred_global_mean_subset.h5ad"),
        adata_real=str(OUT / "real_subset.h5ad"),
        control_pert=CONTROL,
        pert_col="perturbation",
        num_threads=2,
        outdir=str(OUT / "run"),
    )
    results, agg = ev.compute()
    # cell-eval 0.8.2 returns polars frames; the evaluator also writes its
    # own results.csv / agg_results.csv into the outdir.
    if hasattr(results, "write_csv"):
        results.write_csv(OUT / "cell_eval_results.csv")
        agg.write_csv(OUT / "cell_eval_agg_results.csv")
    else:
        results.to_csv(OUT / "cell_eval_results.csv", index=False)
        agg.to_csv(OUT / "cell_eval_agg_results.csv", index=False)
    print("cell-eval aggregate:")
    print(agg.to_string())
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
