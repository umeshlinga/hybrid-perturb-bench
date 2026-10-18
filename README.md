# hybrid-perturb-bench — Build 2 (Weeks 4–7, starts after Build 1 ships)

Hybrid virtual-cell perturbation benchmark — "beat the mean, honestly."

Status: **planned, not built yet.** Scaffolded from the 12-week Flagship Roadmap (2026-10-02).

Plan: perturbation-held-out splits on pertpy/scPerturb / Tahoe-100M slice (stream 100k–1M cells, never download whole) → mean + linear baselines first → GEARS comparator → winners' hybrid (pseudobulk delta + ESM-2 embeddings + DEG-frequency/statistical features + small MLP) → Arc cell-eval (PDS/DES/Pearson-delta/MAE) → public leaderboard including where everything collapses to the mean. Stretch W7: hold out an entire cell line (2026 VCC zero-shot context).

Licence note: Arc State is CC BY-NC-SA / non-commercial model — comparator only.
