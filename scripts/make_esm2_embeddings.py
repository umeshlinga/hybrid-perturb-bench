"""ESM-2 embeddings for the 105 single-perturbation genes.

Model: esm2_t6_8M_UR50D (6 layers, 8M params, 320-d) via fair-esm,
weights from the official fair-esm release URL. Sequences: canonical
human proteins fetched from UniProt (scripts/fetch_uniprot.py →
data/uniprot_sequences.json). Embedding = mean of last-layer token
representations over residues (BOS/EOS excluded). Sequences longer than
1022 residues are truncated to the first 1022 (the standard ESM input
convention); every truncation is logged.

Output: results/esm2_embeddings.npz (genes, embeddings) +
results/esm2_embeddings_coverage.json. Genes without a UniProt sequence
get NO row here — the hybrid treats them as missing (neutral block) and
RESULTS.md states the coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MAX_LEN = 1022


def main() -> None:
    import esm

    seqs = json.loads((ROOT / "data" / "uniprot_sequences.json").read_text())
    have = {g: v for g, v in seqs.items() if v.get("status") == "ok"}
    missing = sorted(g for g, v in seqs.items() if v.get("status") != "ok")
    print(f"genes with sequence: {len(have)}; without: {missing}", flush=True)

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    layer = model.num_layers

    genes = sorted(have)
    truncated = []
    embs: dict[str, np.ndarray] = {}
    BATCH = 4
    for i in range(0, len(genes), BATCH):
        chunk = genes[i : i + BATCH]
        data = []
        for g in chunk:
            s = have[g]["sequence"]
            if len(s) > MAX_LEN:
                truncated.append((g, len(s)))
                s = s[:MAX_LEN]
            data.append((g, s))
        _, _, tokens = batch_converter(data)
        with torch.no_grad():
            out = model(tokens, repr_layers=[layer], return_contacts=False)
        reps = out["representations"][layer]
        for j, g in enumerate(chunk):
            L = min(len(have[g]["sequence"]), MAX_LEN)
            vec = reps[j, 1 : L + 1].mean(dim=0).numpy().astype(np.float32)
            embs[g] = vec
        print(f"{min(i + BATCH, len(genes))}/{len(genes)} embedded", flush=True)

    gene_arr = np.array(sorted(embs))
    mat = np.stack([embs[g] for g in gene_arr])
    np.savez(ROOT / "results" / "esm2_embeddings.npz", genes=gene_arr, embeddings=mat)
    coverage = {
        "model": "esm2_t6_8M_UR50D",
        "embedding_dim": int(mat.shape[1]),
        "n_genes_embedded": int(len(gene_arr)),
        "genes_without_sequence": missing,
        "truncated_sequences": [{"gene": g, "length": L} for g, L in truncated],
        "pooling": "mean of last-layer token representations, BOS/EOS excluded",
        "max_len": MAX_LEN,
    }
    (ROOT / "results" / "esm2_embeddings_coverage.json").write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
