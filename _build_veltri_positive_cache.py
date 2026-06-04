# Build the veltri_positive parents + mutants cache from results/data/all_in,
# mirroring the existing veltri_negative cache layout.
import time
from pathlib import Path
import pandas as pd

ROOT = Path(".").resolve()
ALL_IN = ROOT / "results" / "data" / "all_in"
CACHE = ALL_IN / "_cache"
CACHE.mkdir(exist_ok=True)

TAG = "veltri_positive"
parents_out = CACHE / f"parents_hydramp_{TAG}.parquet"
mutants_out = CACHE / f"mutants_hydramp_{TAG}.parquet"

t0 = time.time()

# --- parents: filter the (small) apex parents file by dataset membership ---
parents = pd.read_csv(ALL_IN / "peptides_apex.csv")
parents = parents[parents["dataset"].str.contains(TAG, na=False)].reset_index(drop=True)
parent_seqs = set(parents["sequence"])
print(f"[{time.time()-t0:.0f}s] positive parents: {len(parents)} rows, "
      f"{len(parent_seqs)} unique sequences")
parents.to_parquet(parents_out, index=False)
print(f"[{time.time()-t0:.0f}s] wrote {parents_out}")

# --- mutants: stream the 14GB apex mutants file, keep rows whose parent is positive ---
chunks = []
kept = 0
total = 0
reader = pd.read_csv(ALL_IN / "peptides_mutants_apex.csv", chunksize=1_000_000)
for ci, chunk in enumerate(reader):
    total += len(chunk)
    sub = chunk[chunk["parent"].isin(parent_seqs)]
    if len(sub):
        chunks.append(sub)
        kept += len(sub)
    if ci % 5 == 0:
        print(f"[{time.time()-t0:.0f}s] chunk {ci}: scanned {total:,}, kept {kept:,}")

mutants = pd.concat(chunks, ignore_index=True) if chunks else parents.iloc[0:0]
print(f"[{time.time()-t0:.0f}s] DONE scan: {total:,} rows scanned, {kept:,} kept")
mutants.to_parquet(mutants_out, index=False)
print(f"[{time.time()-t0:.0f}s] wrote {mutants_out}  shape={mutants.shape}")
