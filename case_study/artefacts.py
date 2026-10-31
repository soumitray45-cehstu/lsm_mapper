"""Block-edge artefact index for susceptibility maps (Reviewer 2, comment 9).

For horizontally and vertically adjacent valid cells, compare the mean absolute
change in susceptibility probability across the edges of a coarse input layer
(cells where the raw layer value changes) with the mean change elsewhere.
Index = mean|dP| at edges / mean|dP| off edges. About 1 means the map has no
extra discontinuity at the coarse layer's block edges; values well above 1
mean the blocks show through ("grid pattern").

    python case_study/artefacts.py OUT.csv EDGE_LAYER:name=path ... -- MAP:name=path ...
"""
import sys

import numpy as np
import pandas as pd
import rasterio


def read(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype(np.float64)
        nd = s.nodata
    if nd is not None and not np.isnan(nd):
        a[a == nd] = np.nan
    a[np.abs(a) > 1e30] = np.nan
    return a


def index(prob, layer):
    out = []
    for ax in (0, 1):
        dp = np.abs(np.diff(prob, axis=ax))
        edge = np.diff(layer, axis=ax) != 0
        ok = np.isfinite(dp) & np.isfinite(np.diff(layer, axis=ax))
        out.append((dp[ok & edge].sum(), (ok & edge).sum(), dp[ok & ~edge].sum(), (ok & ~edge).sum()))
    s = np.array(out).sum(0)
    return (s[0] / s[1]) / (s[2] / s[3]), int(s[1])


if __name__ == "__main__":
    args = sys.argv[2:]
    k = args.index("--")
    layers = dict(a.split("=", 1) for a in args[:k])
    maps = dict(a.split("=", 1) for a in args[k + 1:])
    L = {n: read(p) for n, p in layers.items()}
    rows = []
    for mn, mp in maps.items():
        P = read(mp)
        for ln, la in L.items():
            idx, n = index(P, la)
            rows.append({"map": mn, "edges_of": ln, "artefact_index": idx, "edge_pairs": n})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
