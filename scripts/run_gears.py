"""GEARS comparator (milestone 2) — runs in .venv-gears (cell-gears 0.1.2).

Bounded, documented deviations from a stock GEARS run, all forced by the
CPU-only 2-core / 7 GB environment and recorded here and in RESULTS.md:
  * cells subsampled to at most CAP cells per condition (seed 0);
  * 5 training epochs, batch size 64;
  * validation = 8 of the 84 train singles (model selection only; GEARS
    still trains exclusively on train-perturbation cells — no test
    perturbation is in train or val).
Everything else follows GEARS' own pipeline: per-cell library
normalization (1e4) + log1p before new_data_process (the scale GEARS is
designed for), its GO perturbation graph, and its custom-split API to
reproduce the milestone-1 split exactly:
  train/val conditions: the 84 train singles; test: 21 singles + 131 combos.

Outputs: results/gears_predictions.json {condition_label: {gene: value}}
on GEARS' gene set + results/gears_run.json (coverage, timing, config).
Scoring under the local contract happens in scripts/score_gears.py
(main venv), so this script never invents metrics.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "NormanWeissman2019_filtered.h5ad"
GEARS_DIR = ROOT / "gears_data"
CAP = 128
EPOCHS = 5
BATCH = 64
SEED = 0


def cond_label(pert: str) -> str:
    if pert == "control":
        return "ctrl"
    parts = pert.split("_")
    return parts[0] + "+ctrl" if len(parts) == 1 else "+".join(parts)


def main() -> None:
    t0 = time.time()
    split = json.loads((RESULTS / "split.json").read_text())
    train_perts = split["train_perturbations"]
    test_perts = split["test_perturbations"]
    singles = sorted(set(train_perts) | set(test_perts))

    print("loading h5ad ...", flush=True)
    adata = ad.read_h5ad(DATA)
    pert = adata.obs["perturbation"].astype(str)
    combos = sorted(p for p in pert.unique() if p != "control" and p not in singles)
    adata.obs["condition"] = [cond_label(p) for p in pert]
    adata.obs["cell_type"] = "K562"
    adata.var["gene_name"] = [str(g) for g in adata.var_names]

    rng = np.random.default_rng(SEED)
    keep = []
    for cond, idx in adata.obs.groupby("condition", observed=True).indices.items():
        idx = np.asarray(idx)
        if len(idx) > CAP:
            idx = rng.choice(idx, size=CAP, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    adata = adata[keep].copy()
    print(f"subsampled: {adata.shape[0]} cells, {adata.obs['condition'].nunique()} conditions", flush=True)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    from gears import GEARS, PertData

    pert_data = PertData(str(GEARS_DIR))
    pert_data.new_data_process(dataset_name="norman2019_m2", adata=adata)
    print(f"data process done at {time.time() - t0:.0f}s", flush=True)

    val_perts = sorted(train_perts)[::10][:8]
    train_conds = [p + "+ctrl" for p in train_perts if p not in val_perts]
    val_conds = [p + "+ctrl" for p in val_perts]
    test_conds = [p + "+ctrl" for p in test_perts] + [cond_label(c) for c in combos]
    split_dict = {"train": train_conds, "val": val_conds, "test": test_conds}
    split_path = GEARS_DIR / "custom_split.pkl"
    with open(split_path, "wb") as fh:
        pickle.dump(split_dict, fh)
    pert_data.prepare_split(split="custom", split_dict_path=str(split_path))
    pert_data.get_dataloader(batch_size=BATCH, test_batch_size=BATCH)

    model = GEARS(pert_data, device="cpu")
    model.model_initialize(hidden_size=64)
    t_train = time.time()
    model.train(epochs=EPOCHS)
    print(f"training done in {time.time() - t_train:.0f}s", flush=True)

    gene_names = [str(g) for g in pert_data.gene_names]
    preds: dict[str, dict[str, float]] = {}
    failures: dict[str, str] = {}
    targets = [[p] for p in []]  # placeholder
    singles_targets = [[p, "ctrl"] for p in test_perts]
    combo_targets = [c.split("_") for c in combos]
    for combo in [False, True]:
        for pert_genes in (combo_targets if combo else singles_targets):
            label = "_".join(pert_genes)
            try:
                out, _ = model.predict([pert_genes])
                vec = out["_".join(pert_genes)]
                preds[label] = {g: float(v) for g, v in zip(gene_names, np.asarray(vec).ravel())}
            except Exception as exc:
                failures[label] = str(exc)
            print(f"predicted {label} ({len(preds)} ok, {len(failures)} failed)", flush=True)

    (RESULTS / "gears_predictions.json").write_text(json.dumps(preds))
    info = {
        "cell_gears_version": "0.1.2",
        "cap_cells_per_condition": CAP,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "seed": SEED,
        "val_perturbations": val_perts,
        "n_conditions_predicted": len(preds),
        "prediction_failures": failures,
        "gears_gene_set_size": len(gene_names),
        "eval_genes_covered": None,  # filled by score_gears.py
        "wall_seconds_total": time.time() - t0,
    }
    (RESULTS / "gears_run.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)
    print("DONE GEARS", flush=True)


if __name__ == "__main__":
    main()
