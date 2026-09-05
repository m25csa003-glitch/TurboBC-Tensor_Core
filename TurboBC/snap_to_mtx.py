#!/usr/bin/env python3
"""Convert a SNAP edge list (whitespace .txt or comma .csv) into a TurboBC .mtx.

Same three requirements as edgelist_to_mtx.py:
  * 1-based indices
  * row = source, col = destination -- the BFS step is f <- A'f, so the
    traversal walks column i and reads f[IC[k]] for the sources feeding i
  * entries grouped by column, because CP_RP_dg builds CP from in-degree
    counts while IC is copied verbatim in file order

--undirected emits both (u,v) and (v,u); run those with ug 0.
"""
import sys, gzip, os

src, out = sys.argv[1], sys.argv[2]
undirected = '--undirected' in sys.argv

op = gzip.open if src.endswith('.gz') else open
edges = set()
maxid = 0
with op(src, 'rt') as f:
    for line in f:
        line = line.strip()
        if not line or line[0] == '#':
            continue
        parts = line.replace(',', ' ').split()
        try:
            u, v = int(parts[0]), int(parts[1])
        except ValueError:      # csv header row
            continue
        if u == v:              # TurboBC's BC has no use for self-loops
            continue
        edges.add((u, v))
        if undirected:
            edges.add((v, u))
        maxid = max(maxid, u, v)

n = maxid + 1                   # ids kept as-is; gaps become isolated vertices
entries = sorted((d, s) for s, d in edges)
with open(out, 'w') as f:
    f.write('%%MatrixMarket matrix coordinate pattern general\n')
    f.write('%% converted from %s by snap_to_mtx.py\n' % os.path.basename(src))
    f.write('%d %d %d\n' % (n, n, len(entries)))
    for col, row in entries:
        f.write('%d %d\n' % (row + 1, col + 1))
print('%s: n=%d nz=%d' % (out, n, len(entries)))
