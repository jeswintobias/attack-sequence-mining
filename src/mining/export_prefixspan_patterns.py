"""
Export the top 30 PrefixSpan patterns of length >= 2 by support.

Mines on train sequences only, using the B5 support (1% of dup-weighted
train sequences) and max_len=4. Does not load any test file.
"""

import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "mining")]:
    if p not in sys.path:
        sys.path.insert(0, p)
import config
from prefixspan import prefixspan_mine


def export_top_patterns(results_dir: Path = None, min_support_pct: float = 0.01,
                        max_len: int = 4, top_n: int = 30):
    if results_dir is None:
        results_dir = config.PREFIXSPAN_RESULTS_DIR

    train_df = pd.read_csv(
        config.BASE_DIR / "data" / "processed" / "train_sequences.csv"
    )
    seqs = [tuple(ast.literal_eval(r["sequence"])) for _, r in train_df.iterrows()]
    weights = train_df["dup_count"].tolist()
    total_w = sum(weights)
    min_sup = int(total_w * min_support_pct)

    patterns = prefixspan_mine(
        seqs, min_support=min_sup, max_len=max_len, weights=weights
    )
    rows = []
    for sup, pat in patterns:
        if len(pat) >= 2:
            rows.append({
                "pattern": " ".join(pat),
                "length": len(pat),
                "support": int(sup),
                "support_frac": round(sup / total_w, 6),
            })
    df = pd.DataFrame(rows).head(top_n)
    out = results_dir / "prefixspan_patterns.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} patterns to {out.name} "
          f"(min_sup={min_sup}, total_w={total_w})")
    return df


if __name__ == "__main__":
    export_top_patterns()
