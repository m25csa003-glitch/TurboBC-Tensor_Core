#!/usr/bin/env python3
"""Convert a SuiteSparse 'coordinate ... symmetric' matrix into a TurboBC .mtx.

SuiteSparse stores a symmetric matrix as its lower triangle only, with values.
TurboBC's reader wants the same three things every other converter here
produces:
  * 1-based indices
  * row = source, col = destination -- the BFS step is f <- A'f, so the
    traversal walks column i and reads f[IC[k]] for the sources feeding i
  * entries grouped by column, because CP_RP_dg builds CP from in-degree
    counts while IC is copied verbatim in file order

So each stored entry (i,j) is emitted as both (i,j) and (j,i); diagonal
entries are emitted once.  Values are dropped -- BC is unweighted.  The result
already holds both directions, so run it with --ug 0, not --ug 1.

This is how nd3k.mtx and nd6k.mtx in graphData/ were produced.

Usage:  suitesparse_sym_to_mtx.py  nd6k.mtx  ../graphData/nd6k.mtx
"""
import sys, os

if len(sys.argv) < 3:
    sys.exit(__doc__)
src, out = sys.argv[1], sys.argv[2]

n = None
entries = []
with open(src) as f:
    for line in f:
        if line.startswith('%'):
            continue
        parts = line.split()
        if n is None:
            n = int(parts[0])
            if int(parts[1]) != n:
                sys.exit("not square: %s x %s" % (parts[0], parts[1]))
            continue
        i, j = int(parts[0]), int(parts[1])
        entries.append((j, i))                 # (col, row)
        if i != j:
            entries.append((i, j))             # mirror

entries.sort()                                  # group by column
with open(out, 'w') as f:
    f.write('%%MatrixMarket matrix coordinate pattern general\n')
    f.write('%% expanded from symmetric %s by suitesparse_sym_to_mtx.py\n'
            % os.path.basename(src))
    f.write('%d %d %d\n' % (n, n, len(entries)))
    for col, row in entries:
        f.write('%d %d\n' % (row, col))
print('%s: n=%d nz=%d' % (out, n, len(entries)))
