"""Standalone combo-ceiling + finalization for milestone 2.

The full milestone-2 runner was killed twice during this streaming step
(once by a machine restart, once under memory pressure while another
job ran), after its checkpoint CSVs were already written. This script
does ONLY the remaining work, with a small memory footprint (no cache
pickle, no feature frames):

  1. split-half ceiling for control + the 131 combos (seed 0, same
     construction as milestone 1), appended to the milestone-2 CSVs
     (any existing combo-ceiling rows are replaced);
  2. results/milestone2_run.json (run configuration record);
  3. prints the final aggregate tables + hybrid combo subgroups.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_contract import ceiling_from_pseudobulks
from streaming import normalize_profiles, open_backed, split_half_counts_stream

DATA = ROOT / "data" / "NormanWeissman2019_filtered.h5ad"
RESULTS = ROOT / "results"
SEED = 0
DEG_K = 50
PERT_COL = "perturbation"
CONTROL = "control"


def main() -> None:
    import pickle

    split = json.loads((RESULTS / "split.json").read_text())
    train_perts, test_perts = split["train_perturbations"], split["test_perturbations"]
    with open(RESULTS / "_m1_cache.pkl", "rb") as fh:
        eval_genes = pickle.load(fh)["eval_genes"]

    adata_b = open_backed(DATA)
    labels = adata_b.obs[PERT_COL].astype(str).to_numpy()
    singles = set(train_perts) | set(test_perts)
    combos = sorted(p for p in np.unique(labels) if p != CONTROL and p not in singles)
    print(f"combos: {len(combos)}", flush=True)

    rng = np.random.default_rng(SEED)
    half = np.full(len(labels), -1, dtype=int)
    for pert in [CONTROL] + combos:
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
    ceil_per.insert(0, "eval_block", "combos")
    ceil_agg = ceil_agg.reset_index(names="metric")
    ceil_agg.insert(0, "gene_set", "eval2000")
    ceil_agg.insert(0, "method", "ceiling_split_half")
    ceil_agg.insert(0, "eval_block", "combos")

    per_df = pd.read_csv(RESULTS / "milestone2_metrics_per_perturbation.csv")
    agg_df = pd.read_csv(RESULTS / "milestone2_metrics_aggregate.csv")
    drop_p = (per_df["eval_block"] == "combos") & (per_df["method"] == "ceiling_split_half")
    drop_a = (agg_df["eval_block"] == "combos") & (agg_df["method"] == "ceiling_split_half")
    per_df = pd.concat([per_df[~drop_p], ceil_per], ignore_index=True)
    agg_df = pd.concat([agg_df[~drop_a], ceil_agg], ignore_index=True)
    per_df.to_csv(RESULTS / "milestone2_metrics_per_perturbation.csv", index=False)
    agg_df.to_csv(RESULTS / "milestone2_metrics_aggregate.csv", index=False)
    print("combo ceiling appended", flush=True)

    esm_cov = json.loads((RESULTS / "esm2_embeddings_coverage.json").read_text())
    run_info = {
        "seed": SEED,
        "train_perturbations": len(train_perts),
        "test_perturbations": len(test_perts),
        "n_combos": len(combos),
        "ridge_combo_coverage": int(
            agg_df[(agg_df.eval_block == "combos") & (agg_df.method == "linear_ridge")
                   & (agg_df.metric == "pearson_delta")]["n_perturbations"].iloc[0]
        ),
        "esm_model": esm_cov["model"],
        "esm_genes_available": esm_cov["n_genes_embedded"],
        "esm_missing_genes": esm_cov["genes_without_sequence"],
        "hybrid_config": {
            "delta_pca_components": 40, "esm_pca": 64, "coex_pca": 32,
            "hidden": 128, "dropout": 0.25, "epochs": 500, "lr": 1e-3,
            "weight_decay": 1e-2, "retrieval_topk": 8, "retrieval_temp": 0.1,
        },
        "note": "Full runner killed twice during the combo-ceiling stream "
                "(machine restart, then memory pressure); metrics before the "
                "stream were checkpointed by the runner and the ceiling was "
                "completed by scripts/run_combo_ceiling.py.",
    }
    (RESULTS / "milestone2_run.json").write_text(json.dumps(run_info, indent=2))

    pd.set_option("display.width", 250)
    for block in ["singles", "combos"]:
        print(f"\n=== MILESTONE 2 AGGREGATE means (eval2000, {block}) ===", flush=True)
        sub = agg_df[(agg_df["eval_block"] == block) & (agg_df["gene_set"] == "eval2000")]
        print(sub.pivot_table(index="method", columns="metric", values="mean").round(4).to_string())

    train_set, test_set = set(train_perts), set(test_perts)
    hp = per_df[
        (per_df["eval_block"] == "combos") & (per_df["method"] == "hybrid")
        & (per_df["gene_set"] == "eval2000")
    ].set_index("perturbation")
    ap = per_df[
        (per_df["eval_block"] == "combos") & (per_df["method"] == "additive_hybrid")
        & (per_df["gene_set"] == "eval2000")
    ].set_index("perturbation")
    groups = {
        "both components train singles": [c for c in combos if all(g in train_set for g in c.split("_"))],
        "one train + one test component": [
            c for c in combos
            if any(g in test_set for g in c.split("_")) and any(g in train_set for g in c.split("_"))
        ],
        "both components test singles": [c for c in combos if all(g in test_set for g in c.split("_"))],
    }
    for name, group in groups.items():
        if not group:
            continue
        for label, df in [("hybrid", hp), ("additive_hybrid", ap)]:
            sub = df.loc[group]
            print(
                f"{label} combos [{name}] n={len(sub)}: pearson_delta {sub['pearson_delta'].mean():.4f}, "
                f"deg_overlap {sub['deg_overlap_at_k'].mean():.4f}",
                flush=True,
            )
    print("\nDONE combo ceiling + finalization", flush=True)


if __name__ == "__main__":
    main()
