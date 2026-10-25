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

---

# Milestone 2 — RESULTS

The hybrid model, built per the README plan, plus a larger evaluation:
the same 21 held-out singles under the identical milestone-1 contract,
and the 131 combination perturbations as a compositional-generalization
test. Every number below was produced by the commands in §M2.6 and is
stored in `results/milestone2_metrics_aggregate.csv` /
`results/milestone2_metrics_per_perturbation.csv`. The milestone-1
baselines were recomputed inside the milestone-2 run from the same
cache and reproduce the milestone-1 values exactly (global mean
0.5876, ridge 0.5785, control-mean MSE 0.006534).

## M2.1 The hybrid model (as actually built)

- **Target**: pseudobulk delta on the 2,000 eval genes, parameterized
  as coefficients in a PCA basis of the 84 train deltas (40 components)
  around the train mean delta. All fitted statistics — PCA bases,
  standardizers, the delta basis — use the 84 train singles only.
- **Inputs per perturbed gene** (for a combo, the two component genes'
  blocks are averaged before the MLP):
  1. **Gene embeddings — ESM-2, actually used**: `esm2_t6_8M_UR50D`
     (6-layer, 320-d) via fair-esm, mean-pooled over residues.
     Sequences are the reviewed (Swiss-Prot) human UniProt entries for
     the perturbed genes: coverage **105/105** genes
     (`scripts/fetch_uniprot.py`, `data/uniprot_sequences.json`). 14
     sequences longer than 1,022 residues were truncated to the first
     1,022 (the standard ESM input convention; listed in
     `results/esm2_embeddings_coverage.json`). Standardized, PCA to 64.
  2. **Co-expression block**: the target gene's co-expression profile in
     control cells only (the milestone-1 `feats` frame), standardized,
     PCA to 32. Genes absent from the expression matrix (C19orf26,
     C3orf72) get the neutral (train-mean) block; their ESM-2 block is
     real, since it comes from sequence.
  3. **Statistical / DEG-frequency features**: control expression of
     the target gene, its profile variance across control + train
     perturbations, and summaries of its control-cell co-expression
     profile (mean |r|, max |r|, fraction |r| > 0.3, self-co-expression).
  4. **Pseudobulk-delta retrieval channel**: similarity-weighted mean
     of train perturbations' real pseudobulk deltas (cosine similarity
     in the ESM-2 + co-expression space, top-8, softmax temperature
     0.1), projected onto the delta basis. This channel alone is also
     scored as the `retrieval_knn` reference predictor.
- **MLP**: input → 128 → ReLU → Dropout(0.25) → 128 → ReLU →
  Dropout(0.25) → 40. Adam (lr 1e-3, weight decay 1e-2), full-batch MSE
  on standardized coefficients, 500 epochs, seed 0. All hyperparameters
  fixed a priori; nothing was tuned on test. Final train MSE 0.230
  (standardized-coefficient scale).
- Prediction frames fill non-eval genes with the train mean delta —
  the same convention as the milestone-1 ridge — so all-gene metrics
  are defined for every method.

## M2.2 Headline results — held-out singles (eval2000, mean over test perturbations)

| method | n | pearson_delta | mse_delta | deg_overlap@50 |
|---|---|---|---|---|
| **ceiling (split-half, real data)** | 21 | **0.9167** | **0.001382** | **0.6590** |
| global perturbation mean | 21 | 0.5876 | 0.004324 | 0.3314 |
| linear ridge (co-expression features) | 19 † | 0.5785 | 0.004470 | 0.3326 |
| **hybrid (ESM-2 + co-expression + stats + retrieval → MLP)** | 21 | 0.5366 | 0.005134 | 0.3143 |
| retrieval_knn (hybrid's retrieval channel alone) | 21 | 0.5076 | 0.005692 | 0.2762 |
| control mean (no-change) | 21 | NaN * | 0.006534 | 0.0210 |

\* / † as in milestone 1 (no-change Pearson undefined; ridge cannot
cover perturbations whose target gene is absent from the expression
matrix).

**Reading (interpretation, not measurement):** on held-out single
perturbations the hybrid does **not** beat the baselines it was built
to challenge. It improves on no-change (MSE 0.005134 vs 0.006534) and
on its own retrieval channel (0.5366 vs 0.5076 Pearson), but it sits
below the global mean (0.5876) and the ridge (0.5785) on Pearson Δ, and
below both on DEG overlap (0.3143 vs 0.3314 / 0.3326). With 84 training
perturbations, sequence and co-expression features do not identify
what an unseen single perturbation will do beyond the strong shared
CRISPRa response program that the global mean already captures — the
milestone-1 "collapse to the mean" problem, now measured for a learned
model instead of assumed away. This negative result is reported as-is;
no hyperparameter was adjusted after seeing it.

## M2.3 Combinations — compositional generalization (eval2000, mean over combos)

The 131 two-gene combinations were never used in any training. Four
predictors: global mean and ridge as before (ridge features for a combo
= mean of the two components' co-expression profiles; 129/131 coverage
— the 2 combos containing C19orf26 or C3orf72 have no co-expression
features); the hybrid directly (component feature blocks averaged);
and `additive_hybrid`, the sum of the hybrid's two predicted single
deltas. A split-half ceiling was computed for the combos with the same
construction as milestone 1.

| method | n | pearson_delta | mse_delta | deg_overlap@50 |
|---|---|---|---|---|
| **ceiling (split-half, real data)** | 131 | **0.9314** | **0.004406** | **0.7011** |
| additive_hybrid (sum of predicted single deltas) | 131 | 0.8248 | 0.005471 | 0.5472 |
| **hybrid (direct)** | 131 | 0.7995 | 0.008824 | 0.5098 |
| linear ridge | 129 | 0.5753 | 0.012948 | 0.3033 |
| global perturbation mean | 131 | 0.5663 | 0.013006 | 0.2921 |

Subgroups by whether the combo's components are train or test singles:

| subgroup | n | hybrid pearson | hybrid DEG | additive pearson | additive DEG |
|---|---|---|---|---|---|
| both components train singles | 72 | 0.8253 | 0.5636 | 0.8698 | 0.6253 |
| one train + one test component | 50 | 0.7732 | 0.4584 | 0.7744 | 0.4672 |
| both components test singles | 9 | 0.7387 | 0.3644 | 0.7443 | 0.3667 |

**Reading (interpretation, not measurement):** on combinations the
picture reverses. Both hybrid variants score far above the global mean
and the ridge (Pearson 0.80–0.82 vs 0.57–0.58; DEG overlap 0.51–0.55 vs
0.29–0.30), though still below the real-data ceiling (0.9314 /
0.7011). Two honest qualifications. First, `additive_hybrid`'s
both-train subgroup benefits from the hybrid having been trained on
those singles (its single predictions there are partly in-sample), so
the mixed and both-test subgroups are the cleaner generalization
read — and the hybrid's advantage over the baselines persists there
(0.74–0.77 vs the baselines' 0.57–0.58 overall). Second, combo deltas
are larger in magnitude than single deltas (baseline MSE roughly
triples), and many Norman combos are dominated by one strong
component, which makes per-combo correlation an easier target than
for singles; the combo numbers should not be read as "the model is
better at combos than at singles" without that context. What the
combo test does show is that the per-gene signal the hybrid learned —
too weak to rank unseen singles above the mean — becomes decisive
once responses compose: summing predicted single deltas recovers most
of the combo response, i.e. on this dataset predicted responses
compose close to additively.

Secondary (all 33,694 genes): singles — hybrid pearson_delta 0.5257 /
mse 0.000406 / DEG 0.3086; retrieval_knn 0.4998 / 0.000439 / 0.2676.
Combos — additive_hybrid 0.7460 / 0.000600 / 0.5408; hybrid 0.7204 /
0.000799 / 0.5029; ridge 0.5446 / 0.001043 / 0.2986; global mean
0.5372 / 0.001048 / 0.2873. Medians and per-perturbation values: see
the milestone-2 CSVs.

## M2.4 GEARS comparator — attempted, not completed, no numbers reported

GEARS (Roohani et al.) was installed for real: PyPI's `gears` package
is an unrelated web-asset tool, so GEARS was installed from the
snap-stanford GitHub repository (`cell-gears 0.1.2`) into an isolated
venv (`.venv-gears`), leaving the milestone environment untouched. Its
gene2go and essential-genes files downloaded successfully from Harvard
Dataverse, the dataset was prepared (cells capped at 128 per condition
→ 29,587 cells, 237 conditions; per-cell normalization 1e4 + log1p;
the milestone-1 split reproduced through GEARS' custom-split API, with
8 of the 84 train singles reserved for validation/model selection),
and GEARS' differential-expression processing completed. The run then
stopped inside GEARS' per-cell graph construction
(`create_dataset_file`): after ~43 minutes it had processed 103 of
237 conditions at ~24 s per condition with memory already at ~4.8 GB
of this machine's 7 GB and rising roughly linearly — projected to
exceed available memory before finishing, with multi-epoch CPU
training of ~30k cell graphs still ahead of it. The run was therefore
terminated before training began. **No GEARS predictions exist and no
GEARS numbers are reported anywhere in this document.** The bounded
configuration that was attempted is recorded in
`scripts/run_gears.py`; the partial artifacts remain under
`gears_data/`.

## M2.5 Environment & run history (recorded, none hidden)

- Embeddings: fair-esm 2.0.0 in the main venv; ESM-2 weights from the
  official fair-esm release. GEARS in isolated `.venv-gears`
  (torch 2.14.1+cpu, torch-geometric 2.8.1, cell-gears 0.1.2).
- The first milestone-2 run was killed by a machine restart during the
  combo-ceiling stream, and a second run died during the same stream
  under memory pressure from a concurrent job. The runner was patched
  to checkpoint all metrics to CSV before that stream, and the combo
  ceiling was completed by the memory-light standalone
  `scripts/run_combo_ceiling.py`. Predictions, metrics, and this
  section all come from the checkpointed CSVs plus that finalization
  pass; the hybrid's training is deterministic (seed 0) and reproduced
  identical epoch losses across runs.

## M2.6 Commands run

```bash
.venv/bin/pip install fair-esm
.venv/bin/python scripts/fetch_uniprot.py          # -> data/uniprot_sequences.json (105/105)
.venv/bin/python scripts/make_esm2_embeddings.py   # -> results/esm2_embeddings.npz
.venv/bin/python scripts/run_milestone2.py         # -> milestone2 CSVs (checkpoint), predictions_*.h5ad
.venv/bin/python scripts/run_combo_ceiling.py      # -> combo ceiling rows + milestone2_run.json
python3 -m venv .venv-gears                        # GEARS attempt (see §M2.4)
.venv-gears/bin/pip install "git+https://github.com/snap-stanford/GEARS.git"
.venv-gears/bin/python scripts/run_gears.py        # stopped in graph construction; no outputs scored
```

## M2.7 What milestone 2 does NOT claim

- The hybrid does not beat the global mean or the ridge on held-out
  single perturbations (§M2.2) — the headline singles claim remains
  unmet, and the benchmark's "beat the mean" bar is still open there.
- No GEARS comparison exists (§M2.4); nothing in this milestone should
  be read as a statement about GEARS' accuracy.
- Combos are still the same single dataset and cell line (Norman 2019,
  K562); no cross-cell-line, cross-dataset, or zero-shot claim is made.
  `additive_hybrid`'s both-train subgroup is partly in-sample (§M2.3).
- The hybrid's design and hyperparameters were fixed before its test
  scores were seen and were not revised afterward; any future tuning
  belongs to a later milestone with its own held-out protocol.
