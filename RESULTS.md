# Milestone 1 — RESULTS (2026-10-18)

Evaluation contract + three honest baselines, run end-to-end on a bounded
public perturbation dataset. Every number below was produced by the commands
in §6 and is stored in `results/milestone1_metrics_aggregate.csv` /
`results/milestone1_metrics_per_perturbation.csv`.

## 1. Dataset

- **Norman et al. 2019** Perturb-seq (CRISPRa, K562), GEO **GSE133344**,
  as harmonized by **scPerturb**: file `NormanWeissman2019_filtered.h5ad`.
- Source: Zenodo record 10044268,
  `https://zenodo.org/api/records/10044268/files/NormanWeissman2019_filtered.h5ad/content`
- Size: **698,680,199 bytes** (699 MB); MD5 verified against the Zenodo
  checksum: `c870e6967d91c017d9da827bab183cd6`. Well under the 5 GB cap;
  the Tahoe atlas was not touched.
- Shape: **111,445 cells × 33,694 genes**; 237 perturbation labels
  (control + 105 singles + 131 combos), 11,855 control cells.
- Why it fits: the canonical small Perturb-seq benchmark — single-cell
  context (one cell line), pooled CRISPRa singles + combos, rich phenotype;
  small enough to run honestly on a 2-core / 7 GB box.
- Source note: pertpy's own figshare copy of Norman 2019
  (`figshare.com/ndownloader/files/34027562`) returned HTTP 202 with an
  empty body on repeated attempts (async generation never completed from
  this machine), so the identical dataset was taken from scPerturb/Zenodo
  instead and checksum-verified.

## 2. Split (headline)

- Perturbation column `obs['perturbation']`, control label `control`.
- Kept: control + **single** perturbations (`nperts == 1`) with ≥ 30 cells
  → **105 singles**. Combos (131) deferred to a later milestone.
- **Entire perturbations held out**: seed 0, 20 % → **21 test**
  (BCL2L11, C19orf26, C3orf72, CBFA2T3, CBL, CDKN1B, CEBPB, CKS1B, DLX2,
  FOXA1, FOXF1, FOXL2, GLB1L2, MAPK1, MIDN, PTPN1, SLC4A1, SLC6A9,
  UBASH3A, UBASH3B, ZBTB25) / **84 train**. Full lists in `results/split.json`.
- Profiles: pseudobulk **counts** per perturbation, library-normalized to
  1e4 + log1p after aggregation (standard pseudobulk practice).
- Eval gene set (headline): top **2000** genes by pseudobulk-profile
  variance across control + train perturbations only. All-gene metrics are
  reported as secondary rows in the CSVs.

## 3. Evaluation contract

`src/eval_contract.py`: predicted AnnData + real AnnData (here: pseudobulk
profiles, saved as `results/real_pseudobulk_test.h5ad` and
`results/predictions_<method>.h5ad`) → per-perturbation and aggregate
**mse_delta / rmse_delta, pearson_delta, mse_expr / pearson_expr,
deg_overlap_at_50** (|top-50 |Δ| genes ∩| / 50). Ceiling: disjoint
split-half of each test perturbation's and the control's cells, one half
scored against the other, Pearson Spearman-Brown corrected
(`r' = 2r/(1+r)`) — the same construction as Arc cell-eval's `--ceiling`.
Note: because every prediction frame carries the real control row,
mse_delta and mse_expr coincide for these baselines by construction.

## 4. Headline results — eval2000 gene set, mean over test perturbations

| method | n | pearson_delta | mse_delta | deg_overlap@50 |
|---|---|---|---|---|
| **ceiling (split-half, real data)** | 21 | **0.9167** | **0.001382** | **0.6590** |
| (a) control mean (no-change) | 21 | NaN * | 0.006534 | 0.0210 |
| (b) global perturbation mean | 21 | 0.5876 | 0.004324 | 0.3314 |
| (c) linear ridge (co-expression features) | 19 † | 0.5785 | 0.004470 | 0.3326 |

\* The no-change baseline predicts a delta of exactly 0, so its Pearson
correlation is undefined (zero variance) — it is ranked on MSE / overlap.
† Ridge covers 19/21 test and 83/84 train perturbations: targets KIAA1804
(train), C19orf26 and C3orf72 (test) are absent from the filtered
expression matrix, so no co-expression features exist for them.

Secondary (all 33,694 genes): global mean pearson_delta 0.5683 /
mse_delta 0.000358; ridge 0.5627 / 0.000366; control mean mse 0.000517,
overlap 0.0029. Medians and per-perturbation values: see the CSVs.

**Reading (interpretation, not measurement):** the global mean clearly
beats no-change — Norman CRISPRa perturbations share a strong common
response program. The linear ridge does **not** beat the global mean on
held-out perturbations (slightly worse MSE and Pearson, equal DEG
overlap): with control-only co-expression features it essentially
collapses to the mean. Both sit far below the real-data ceiling
(0.59 vs 0.92 Pearson) — the headroom a real model (GEARS next, then the
hybrid) has to earn, and exactly the "beat the mean, honestly" gap this
benchmark exists to quantify.

## 5. Cross-check against Arc cell-eval 0.8.2

cell-eval installed cleanly, so the global-mean baseline was also scored
with Arc's own `MetricsEvaluator` at cell level on a bounded subset
(control subsampled to 3,000 cells + first 5 test perturbations,
2000 eval genes; `scripts/run_cell_eval_check.py`, outputs in
`results/cell_eval_check/run/`):

| metric (mean over the same 5 perts) | local contract | Arc cell-eval |
|---|---|---|
| pearson_delta | 0.5014 | 0.5182 |
| mse_delta | 0.00280 | 0.00415 |
| overlap@50 | 0.300 | 0.160 |

Per-perturbation pearson_delta tracks closely (C19orf26 0.577 vs 0.577;
C3orf72 0.770 vs 0.780; CBFA2T3 0.524 vs 0.524; CBL 0.683 vs 0.684;
BCL2L11 ≈ 0 in both — the shared failure case). The MSE and overlap gaps
are definitional, not errors: cell-eval computes errors per cell (includes
within-group variance) and its overlap uses pdex DE calls, whereas the
local contract scores pseudobulk means and top-50 |Δ| sets.

## 6. Commands run

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas scikit-learn anndata scanpy matplotlib scipy
.venv/bin/pip install cell-eval pertpy
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
curl -L -o data/NormanWeissman2019_filtered.h5ad \
  https://zenodo.org/api/records/10044268/files/NormanWeissman2019_filtered.h5ad/content
.venv/bin/python scripts/run_milestone1.py        # -> results/*.csv, split.json, *.h5ad
.venv/bin/python scripts/run_cell_eval_check.py   # -> results/cell_eval_check/run/
```

## 7. Environment & blockers encountered (all recorded, none hidden)

- Isolated venv `.venv/` (Python 3.12.3; no conda on this machine).
  Import-verified: anndata 0.13.4, scanpy 1.12.4, scikit-learn 1.9.1,
  numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, **torch 2.14.1+cpu**,
  **cell-eval 0.8.2**, **pertpy 1.4.0**. Full stack from `environment.yml`
  is installed; exact pins in `requirements.txt`.
- **Blocker 1 (worked around openly):** loading the full matrix in memory
  + scanpy `normalize_total`/`log1p` was OOM-killed twice on this 7 GB
  machine (process died with no traceback at the normalization step).
  The pipeline was rewritten to stream the backed h5ad in row chunks
  (`src/streaming.py`) and aggregate pseudobulk counts before
  normalization. No metric definition changed because of this.
- **Blocker 2:** pertpy/figshare download returned HTTP 202 empty bodies
  repeatedly → dataset taken from scPerturb/Zenodo instead (§1).
- **Bug found & fixed mid-run:** an indicator-matrix indexing error in
  the split-half streaming code (`axis 1 index exceeds matrix dimension`)
  failed the first full run at the ceiling step; fixed, and streamed
  intermediates cached (`results/_m1_cache.pkl`, git-ignored) so re-runs
  are cheap. Details in git history.
- Runtime: each streaming pass over the backed file takes ~5–8 min on
  2 cores; the full milestone run ≈ 20 min wall-clock.

## 8. What milestone 1 does NOT claim

- Singles only, one cell line (K562), one dataset. Combos, any cross-cell-
  line / zero-shot claim, GEARS, and the hybrid are later milestones.
- The ridge's features use control cells only; alpha = 100 was fixed a
  priori and never tuned on test.
