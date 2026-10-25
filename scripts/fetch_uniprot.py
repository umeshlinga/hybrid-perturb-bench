"""Fetch canonical human protein sequences from UniProt for the 105 single
perturbation genes (84 train + 21 test from results/split.json).

Strategy per gene (fixed a priori): reviewed (Swiss-Prot) human entry with
an exact gene-name match; if none, an unreviewed human entry with an exact
gene-name match. The first entry returned by UniProt (its own ordering) is
taken. Every lookup outcome is recorded so coverage is auditable; genes with
no sequence are NOT silently dropped — they are listed in the output JSON
with status "missing" and the hybrid falls back to a neutral embedding for
them (documented in RESULTS.md).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "uniprot_sequences.json"
API = "https://rest.uniprot.org/uniprotkb/search"


def query(gene: str, reviewed: bool) -> dict | None:
    q = f"(gene_exact:{gene}) AND (organism_id:9606) AND (reviewed:{str(reviewed).lower()})"
    params = urllib.parse.urlencode(
        {"query": q, "format": "json", "size": 1, "fields": "accession,id,gene_names,sequence"}
    )
    req = urllib.request.Request(f"{API}?{params}", headers={"User-Agent": "hybrid-perturb-bench"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    results = data.get("results", [])
    return results[0] if results else None


def main() -> None:
    split = json.loads((ROOT / "results" / "split.json").read_text())
    genes = sorted(set(split["train_perturbations"]) | set(split["test_perturbations"]))
    print(f"{len(genes)} genes to fetch", flush=True)
    out: dict[str, dict] = {}
    if OUT.exists():
        out = json.loads(OUT.read_text())
        print(f"resuming: {len(out)} already fetched", flush=True)
    for gene in genes:
        if gene in out and out[gene].get("sequence"):
            continue
        entry = None
        reviewed = True
        try:
            entry = query(gene, True)
            if entry is None:
                reviewed = False
                entry = query(gene, False)
        except Exception as exc:  # network hiccup: record, move on
            out[gene] = {"status": f"error: {exc}"}
            print(f"{gene}: ERROR {exc}", flush=True)
            continue
        if entry is None:
            out[gene] = {"status": "missing"}
            print(f"{gene}: MISSING", flush=True)
        else:
            out[gene] = {
                "status": "ok",
                "accession": entry.get("primaryAccession"),
                "reviewed": reviewed,
                "sequence": entry["sequence"]["value"],
                "length": entry["sequence"]["length"],
            }
            print(f"{gene}: {entry.get('primaryAccession')} len={entry['sequence']['length']} reviewed={reviewed}", flush=True)
        OUT.write_text(json.dumps(out, indent=1))
        time.sleep(0.15)
    ok = sum(1 for v in out.values() if v.get("status") == "ok")
    print(f"coverage: {ok}/{len(genes)} genes with a UniProt sequence")
    missing = [g for g, v in out.items() if v.get("status") != "ok"]
    print("without sequence:", missing)


if __name__ == "__main__":
    main()
