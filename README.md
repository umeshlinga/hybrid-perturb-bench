# hybrid-perturb-bench — Build 2 (Weeks 4–7, starts after Build 1 ships)

Hybrid virtual-cell perturbation benchmark — "beat the mean, honestly."

Status: **Milestone 1 done (2026-10-18)** — evaluation contract + three honest baselines run end-to-end on Norman 2019 (K562 CRISPRa, perturbation-held-out split). See RESULTS.md for the numbers. Scaffolded from the 12-week Flagship Roadmap (2026-10-02); next: GEARS comparator, then combos, then the winners' hybrid.

Plan: perturbation-held-out splits on pertpy/scPerturb / Tahoe-100M slice (stream 100k–1M cells, never download whole) → mean + linear baselines first → GEARS comparator → winners' hybrid (pseudobulk delta + ESM-2 embeddings + DEG-frequency/statistical features + small MLP) → Arc cell-eval (PDS/DES/Pearson-delta/MAE) → public leaderboard including where everything collapses to the mean. Stretch W7: hold out an entire cell line (2026 VCC zero-shot context).

Licence note: Arc State is CC BY-NC-SA / non-commercial model — comparator only.
