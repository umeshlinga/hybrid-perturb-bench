"""Score GEARS predictions under the milestone-1 contract (main venv).

Reads results/gears_predictions.json (expression vectors on GEARS' own
gene set, produced by scripts/run_gears.py in .venv-gears), builds
prediction frames on the full gene set — genes GEARS did not predict
fall back to the train mean delta, the same convention the ridge and
hybrid frames use for non-eval genes — and appends method="gears" rows
to the milestone-2 metric CSVs. Also records eval-gene coverage back
into results/gears_run.json.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_contract import aggregate, per_perturbation_metrics

RESULTS = ROOT / "results"
CONTROL = "control"
DEG_K = 50


def build_frame(preds: dict, perts: list[str], real_pb, train_perts) -> pd.DataFrame:
    ctrl = real_pb.loc[CONTROL]
    global_delta = (real_pb.loc[train_perts].to_numpy() - ctrl.to_numpy()).mean(axis=0)
    gene_pos = {g: i for i, g in enumerate(real_pb.columns)}
    frame = pd.DataFrame(index=[CONTROL] + list(perts), columns=real_pb.columns, dtype=float)
    frame.loc[CONTROL] = ctrl.to_numpy()
    for p in perts:
        full = ctrl.to_numpy() + global_delta  # fallback everywhere
        for g, v in preds[p].items():
            if g in gene_pos:
                full[gene_pos[g]] = v
        frame.loc[p] = full
    return frame


def main() -> None:
    split = json.loads((RESULTS / "split.json").read_text())
    train_perts, test_perts = split["train_perturbations"], split["test_perturbations"]
    with open(RESULTS / "_m1_cache.pkl", "rb") as fh:
        cache = pickle.load(fh)
    real_pb, eval_genes = cache["real_pb"], cache["eval_genes"]
    singles = set(train_perts) | set(test_perts)
    combos = sorted(p for p in real_pb.index if p != CONTROL and p not in singles)

    preds = json.loads((RESULTS / "gears_predictions.json").read_text())
    covered = set.intersection(*[set(v) for v in preds.values()]) if preds else set()
    n_eval_covered = len([g for g in eval_genes if g in covered])
    print(f"GEARS predictions: {len(preds)} conditions; eval genes covered: {n_eval_covered}/2000")

    per_rows, agg_rows = [], []
    blocks = {"singles": [p for p in test_perts if p in preds],
              "combos": [c for c in combos if c in preds]}
    for block, perts in blocks.items():
        if not perts:
            continue
        frame = build_frame(preds, perts, real_pb, train_perts)
        for gene_set_name, gene_list in [("eval2000", eval_genes), ("all_genes", None)]:
            m = per_perturbation_metrics(frame, real_pb, CONTROL, DEG_K, gene_list)
            m = m.reset_index(names="perturbation")
            m.insert(0, "gene_set", gene_set_name)
            m.insert(0, "method", "gears")
            m.insert(0, "eval_block", block)
            per_rows.append(m)
            a = aggregate(m.drop(columns=["eval_block", "method", "gene_set", "perturbation"]))
            a = a.reset_index(names="metric")
            a.insert(0, "gene_set", gene_set_name)
            a.insert(0, "method", "gears")
            a.insert(0, "eval_block", block)
            agg_rows.append(a)

    per_df = pd.read_csv(RESULTS / "milestone2_metrics_per_perturbation.csv")
    agg_df = pd.read_csv(RESULTS / "milestone2_metrics_aggregate.csv")
    per_df = per_df[per_df["method"] != "gears"]
    agg_df = agg_df[agg_df["method"] != "gears"]
    pd.concat([per_df] + per_rows, ignore_index=True).to_csv(
        RESULTS / "milestone2_metrics_per_perturbation.csv", index=False
    )
    pd.concat([agg_df] + agg_rows, ignore_index=True).to_csv(
        RESULTS / "milestone2_metrics_aggregate.csv", index=False
    )

    info_path = RESULTS / "gears_run.json"
    info = json.loads(info_path.read_text())
    info["eval_genes_covered"] = n_eval_covered
    info_path.write_text(json.dumps(info, indent=2))

    out = pd.concat(agg_rows)
    print(
        out[out["gene_set"] == "eval2000"]
        .pivot_table(index=["eval_block", "method"], columns="metric", values="mean")
        .round(4)
        .to_string()
    )
    print("DONE scoring GEARS")


if __name__ == "__main__":
    main()
