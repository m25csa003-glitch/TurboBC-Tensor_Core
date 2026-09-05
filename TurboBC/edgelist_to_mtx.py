#!/usr/bin/env python3
"""
Convert a NetworkX-style .edgelist file to a Matrix Market (.mtx) file
readable by TurboBC.

Input line format:   <src> <dst> {'weight': <float>}
                     (the attribute dict is optional)

TurboBC requirements handled here:
  * Matrix Market is 1-based, so every node id is shifted by +1.
  * Entry (row=I, col=J) means the edge I -> J, because the BFS step is
    f <- A'f  (see spmv_seq_ug_csc in spmv_seq.c).
  * For directed graphs (--ug 0) main() copies the file entries straight
    into CscA.IC and builds CP from the in-degree vector, so the entries
    must be grouped by column.  We sort by (column, row).
"""

import argparse
import ast
import sys


def parse(path, weighted):
    edges = []
    max_id = -1
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            src, dst, rest = line.split(None, 2) if '{' in line else \
                             (line.split() + [None] * 3)[:3]
            src, dst = int(src), int(dst)
            w = 1.0
            if weighted and rest:
                attrs = ast.literal_eval(rest)
                if 'weight' not in attrs:
                    sys.exit("line %d: no 'weight' key in %r" % (lineno, rest))
                w = float(attrs['weight'])
            edges.append((src, dst, w))
            max_id = max(max_id, src, dst)
    return edges, max_id + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('infile')
    ap.add_argument('outfile')
    ap.add_argument('--unweighted', action='store_true',
                    help='drop weights and emit a 2-column pattern matrix')
    ap.add_argument('--symmetrize', action='store_true',
                    help='add the reverse of every edge (makes the matrix '
                         'symmetric); NOT for use with TurboBC --ug 1, which '
                         'wants a single triangle only')
    args = ap.parse_args()

    weighted = not args.unweighted
    edges, n = parse(args.infile, weighted)

    if args.symmetrize:
        seen = {(s, d) for s, d, _ in edges}
        edges += [(d, s, w) for s, d, w in edges if (d, s) not in seen]

    # deduplicate, keeping the first weight seen for a repeated edge
    dedup = {}
    for s, d, w in edges:
        dedup.setdefault((s, d), w)

    # sort by column, then row -- this is the CSC ordering TurboBC assumes
    entries = sorted(((d, s, w) for (s, d), w in dedup.items()))

    banner = 'real' if weighted else 'pattern'
    with open(args.outfile, 'w') as f:
        f.write('%%%%MatrixMarket matrix coordinate %s general\n' % banner)
        f.write('%% converted from %s by edgelist_to_mtx.py\n' % args.infile)
        f.write('%% row = source vertex, column = destination vertex\n' % ())
        f.write('%d %d %d\n' % (n, n, len(entries)))
        for col, row, w in entries:
            if weighted:
                f.write('%d %d %.17g\n' % (row + 1, col + 1, w))
            else:
                f.write('%d %d\n' % (row + 1, col + 1))

    print('wrote %s: N = %d, nz = %d, %s'
          % (args.outfile, n, len(entries),
             'weighted' if weighted else 'unweighted'))
    if len(dedup) != len(edges):
        print('note: dropped %d duplicate edge(s)' % (len(edges) - len(dedup)))


if __name__ == '__main__':
    main()
