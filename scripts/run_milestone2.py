"""Milestone 2 runner: the hybrid model + GEARS slot + combo evaluation.

Everything is scored under the EXACT milestone-1 contract (same split
from results/split.json, same top-2000 eval genes, same metrics via
src/eval_contract.py):

  Singles (21 held-out test perturbations):
    ceiling (reused from the milestone-1 run), control mean, global
    mean, linear ridge (recomputed from the milestone-1 cache — values
    must reproduce milestone 1), retrieval_knn (the hybrid's retrieval
    channel alone), hybrid (MLP over ESM-2 + co-expression + statistical
    features + retrieval channel).

  Combinations (131, never used in any training):
    ceiling (fresh split-half stream), global mean, linear ridge
    (features = mean of the two components' co-expression profiles;
    combos with a component absent from the expression matrix are
    excluded for ridge only), additive_hybrid (sum of the hybrid's two
    predicted single deltas), hybrid (direct: component feature blocks
    averaged).

Writes results/milestone2_metrics_{aggregate,per_perturbation}.csv,
results/predictions_*.h5ad, results/milestone2_run.json.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import predict_control_mean, predict_global_mean, predict_linear_ridge
from eval_contract import aggregate, ceiling_from_pseudobulks, per_perturbation_metrics
from hybrid import HybridFeatures, predict_delta, to_full_frame, train_hybrid
from streaming import normalize_profiles, open_backed, split_half_counts_stream

DATA = ROOT / "data" / "NormanWeissman2019_filtered.h5ad"
RESULTS = ROOT / "results"
SEED = 0
DEG_K = 50
PERT_COL = "perturbation"
CONTROL = "control"


def to_anndata(pb: pd.DataFrame, perts: list[str]) -> ad.AnnData:
    sub = pb.loc[[CONTROL] + list(perts)]
    out = ad.AnnData(X=sub.to_numpy(dtype=np.float32))
    out.obs_names = sub.index.astype(str)
    out.var_names = sub.columns.astype(str)
    out.obs["perturbation"] = sub.index.astype(str)
    return out


def score_block(preds: dict[str, pd.DataFrame], real_pb, eval_genes, label: str):
    """Score prediction frames; returns (per_df, agg_df) for eval2000 + all_genes."""
    per_rows, agg_rows = [], []
    for name, pb in preds.items():
        for gene_set_name, gene_list in [("eval2000", eval_genes), ("all_genes", None)]:
            m = per_perturbation_metrics(pb, real_pb, CONTROL, DEG_K, gene_list)
            m = m.reset_index(names="perturbation")
            m.insert(0, "gene_set", gene_set_name)
            m.insert(0, "method", name)
            m.insert(0, "eval_block", label)
            per_rows.append(m)
            a = aggregate(m.drop(columns=["eval_block", "method", "gene_set", "perturbation"]))
            a = a.reset_index(names="metric")
            a.insert(0, "gene_set", gene_set_name)
            a.insert(0, "method", name)
            a.insert(0, "eval_block", label)
            agg_rows.append(a)
    return per_rows, agg_rows


def main() -> None:
    split = json.loads((RESULTS / "split.json").read_text())
    train_perts: list[str] = split["train_perturbations"]
    test_perts: list[str] = split["test_perturbations"]
    ridge_train: list[str] = split["ridge_train_perturbations"]

    with open(RESULTS / "_m1_cache.pkl", "rb") as fh:
        cache = pickle.load(fh)
    real_pb, eval_genes, feats = cache["real_pb"], cache["eval_genes"], cache["feats"]
    singles = set(train_perts) | set(test_perts)
    combos = sorted(p for p in real_pb.index if p != CONTROL and p not in singles)
    print(f"train {len(train_perts)} / test {len(test_perts)} / combos {len(combos)}", flush=True)

    # --- ESM-2 embeddings -----------------------------------------------------
    esm: dict[str, np.ndarray] = {}
    esm_path = RESULTS / "esm2_embeddings.npz"
    if esm_path.exists():
        z = np.load(esm_path)
        esm = {g: z["embeddings"][i] for i, g in enumerate(z["genes"])}
        print(f"ESM-2 embeddings loaded: {len(esm)} genes x {len(next(iter(esm.values())))} dims", flush=True)
    else:
        print("WARNING: no ESM-2 embeddings file — hybrid will use co-expression embeddings only", flush=True)

    # --- Train the hybrid ------------------------------------------------------
    print("Building hybrid features ...", flush=True)
    hf = HybridFeatures(real_pb, feats, eval_genes, train_perts, esm, CONTROL)
    print("Training hybrid MLP ...", flush=True)
    model = train_hybrid(hf)

    # --- Singles predictions ----------------------------------------------------
    hybrid_test = {p: predict_delta(hf, model, [p]) for p in test_perts}
    retr_test = {p: hf.retrieval_delta([p]) for p in test_perts}
    hybrid_all_singles = {p: predict_delta(hf, model, [p]) for p in sorted(singles)}

    preds_singles = {
        "control_mean": predict_control_mean(real_pb, test_perts, CONTROL),
        "global_mean": predict_global_mean(real_pb, train_perts, test_perts, CONTROL),
        "linear_ridge": predict_linear_ridge(
            real_pb, ridge_train, split["ridge_test_perturbations"], feats, CONTROL, genes=eval_genes
        ),
        "retrieval_knn": to_full_frame(real_pb, eval_genes, retr_test, train_perts, CONTROL),
        "hybrid": to_full_frame(real_pb, eval_genes, hybrid_test, train_perts, CONTROL),
    }
    to_anndata(preds_singles["hybrid"], test_perts).write_h5ad(RESULTS / "predictions_hybrid.h5ad")
    to_anndata(preds_singles["retrieval_knn"], test_perts).write_h5ad(
        RESULTS / "predictions_retrieval_knn.h5ad"
    )

    # --- Combo predictions -------------------------------------------------------
    combo_genes = {c: c.split("_") for c in combos}
    hybrid_combo = {c: predict_delta(hf, model, gs) for c, gs in combo_genes.items()}
    additive_combo = {
        c: hybrid_all_singles[gs[0]] + hybrid_all_singles[gs[1]] for c, gs in combo_genes.items()
    }
    ridge_combos = [c for c in combos if all(g in feats.index for g in combo_genes[c])]
    combo_feats = feats.copy()
    for c in ridge_combos:
        combo_feats.loc[c] = feats.loc[combo_genes[c]].mean(axis=0)
    preds_combos = {
        "global_mean": predict_global_mean(real_pb, train_perts, combos, CONTROL),
        "linear_ridge": predict_linear_ridge(
            real_pb, ridge_train, ridge_combos, combo_feats, CONTROL, genes=eval_genes
        ),
        "additive_hybrid": to_full_frame(real_pb, eval_genes, additive_combo, train_perts, CONTROL),
        "hybrid": to_full_frame(real_pb, eval_genes, hybrid_combo, train_perts, CONTROL),
    }
    to_anndata(preds_combos["hybrid"], combos).write_h5ad(RESULTS / "predictions_hybrid_combos.h5ad")
    to_anndata(preds_combos["additive_hybrid"], combos).write_h5ad(
        RESULTS / "predictions_additive_hybrid_combos.h5ad"
    )
    to_anndata(real_pb, combos).write_h5ad(RESULTS / "real_pseudobulk_combos.h5ad")
    print(f"ridge combo coverage: {len(ridge_combos)}/{len(combos)}", flush=True)

    # --- Score singles + combos ---------------------------------------------------
    per_rows, agg_rows = [], []
    p, a = score_block(preds_singles, real_pb, eval_genes, "singles")
    per_rows += p; agg_rows += a
    p, a = score_block(preds_combos, real_pb, eval_genes, "combos")
    per_rows += p; agg_rows += a

    # --- Singles ceiling: reuse milestone-1 aggregates -----------------------------
    m1_agg = pd.read_csv(RESULTS / "milestone1_metrics_aggregate.csv")
    ceil = m1_agg[(m1_agg["method"] == "ceiling_split_half") & (m1_agg["gene_set"] == "eval2000")]
    ceil_rows = ceil.assign(eval_block="singles", method="ceiling_split_half")[
        ["eval_block", "method", "gene_set", "metric", "mean", "median", "n_perturbations"]
    ]
    agg_rows.append(ceil_rows)
    m1_per = pd.read_csv(RESULTS / "milestone1_metrics_per_perturbation.csv")
    ceil_per = m1_per[(m1_per["method"] == "ceiling_split_half")].assign(eval_block="singles")
    per_rows.append(ceil_per[["eval_block", "method", "gene_set", "perturbation",
                              "mse_delta", "rmse_delta", "pearson_delta", "mse_expr",
                              "pearson_expr", "deg_overlap_at_k"]])

    # Checkpoint BEFORE the long ceiling stream (a previous run was lost
    # to a machine restart at exactly this point).
    pd.concat(per_rows, ignore_index=True).to_csv(
        RESULTS / "milestone2_metrics_per_perturbation.csv", index=False
    )
    pd.concat(agg_rows, ignore_index=True).to_csv(
        RESULTS / "milestone2_metrics_aggregate.csv", index=False
    )
    print("checkpoint: CSVs written before combo-ceiling stream", flush=True)

    # --- Combo ceiling: fresh split-half stream --------------------------------------
    print("Streaming split-half ceiling for combos ...", flush=True)
    adata_b = open_backed(DATA)
    labels = adata_b.obs[PERT_COL].astype(str).to_numpy()
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
    ceil_per_c, ceil_agg_c = ceiling_from_pseudobulks(pb_a, pb_b, CONTROL, DEG_K, eval_genes)
    ceil_per_c = ceil_per_c.reset_index(names="perturbation")
    ceil_per_c.insert(0, "gene_set", "eval2000")
    ceil_per_c.insert(0, "method", "ceiling_split_half")
    ceil_per_c.insert(0, "eval_block", "combos")
    per_rows.append(ceil_per_c)
    ceil_agg_c = ceil_agg_c.reset_index(names="metric")
    ceil_agg_c.insert(0, "gene_set", "eval2000")
    ceil_agg_c.insert(0, "method", "ceiling_split_half")
    ceil_agg_c.insert(0, "eval_block", "combos")
    agg_rows.append(ceil_agg_c)

    per_df = pd.concat(per_rows, ignore_index=True)
    agg_df = pd.concat(agg_rows, ignore_index=True)
    per_df.to_csv(RESULTS / "milestone2_metrics_per_perturbation.csv", index=False)
    agg_df.to_csv(RESULTS / "milestone2_metrics_aggregate.csv", index=False)
    print("checkpoint: milestone2 CSVs written (incl. combo ceiling)", flush=True)

    run_info = {
        "seed": SEED,
        "train_perturbations": len(train_perts),
        "test_perturbations": len(test_perts),
        "n_combos": len(combos),
        "ridge_combo_coverage": len(ridge_combos),
        "esm_genes_available": len(esm),
        "esm_missing_genes": sorted(set(train_perts + test_perts) - set(esm)),
        "hybrid_config": {
            "delta_pca_components": 40, "esm_pca": 64, "coex_pca": 32,
            "hidden": 128, "dropout": 0.25, "epochs": 500, "lr": 1e-3,
            "weight_decay": 1e-2, "retrieval_topk": 8, "retrieval_temp": 0.1,
        },
    }
    (RESULTS / "milestone2_run.json").write_text(json.dumps(run_info, indent=2))

    pd.set_option("display.width", 250)
    for block in ["singles", "combos"]:
        print(f"\n=== MILESTONE 2 AGGREGATE means (eval2000, {block}) ===", flush=True)
        sub = agg_df[(agg_df["eval_block"] == block) & (agg_df["gene_set"] == "eval2000")]
        print(
            sub.pivot_table(index="method", columns="metric", values="mean")
            .round(4)
            .to_string()
        )
    # Hybrid combo subgroups
    hp = per_df[
        (per_df["eval_block"] == "combos")
        & (per_df["method"] == "hybrid")
        & (per_df["gene_set"] == "eval2000")
    ].set_index("perturbation")
    train_set = set(train_perts)
    both = [c for c in combos if all(g in train_set for g in combo_genes[c])]
    hast = [c for c in combos if c not in both]
    for name, group in [("both components train singles", both), ("has a test-single component", hast)]:
        sub = hp.loc[group]
        print(
            f"hybrid combos [{name}] n={len(sub)}: pearson_delta {sub['pearson_delta'].mean():.4f}, "
            f"deg_overlap {sub['deg_overlap_at_k'].mean():.4f}",
            flush=True,
        )
    print("\nDONE. Milestone-2 results in", RESULTS, flush=True)


if __name__ == "__main__":
    main()
