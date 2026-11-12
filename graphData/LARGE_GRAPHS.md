# amazon.mtx — Fetch and Convert

`amazon.mtx` is 25 MB, large enough that it is kept outside the repository and
`.gitignore`d; `graphData/` holds a symlink to it. Everything else in
`graphData/` is committed normally.

| File | Nodes | Edges | Density | Source |
|:---|---:|---:|---:|:---|
| `amazon.mtx` | 548,552 | 1,851,744 | 0.00062% | SNAP `com-Amazon` |

The node count is `max_id + 1`, so the ID gaps in the SNAP file appear as
213,689 isolated vertices. This matches how the other SNAP graphs here
(`epinions`, `fb_artist`, `fb_new_sites`) were prepared.

## Command

Run from the `TurboBC/` source directory:

```bash
wget https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz
python3 snap_to_mtx.py com-amazon.ungraph.txt.gz ../graphData/amazon.mtx --undirected
```

Verify:

```bash
grep -v '^%' ../graphData/amazon.mtx | head -1
# expected:  548552 548552 1851744
```

---

## Graphs That Were Tested and Removed

Three larger graphs were added, benchmarked, and then dropped from the suite.
They are documented here because the reason is a real finding, not a
convenience: **all three drive the shortest-path counts past what a 32-bit
integer can hold**, and the original TurboBC design accumulates the frontier in
`int`. See "int32 overflow in the path counts" in `REPRODUCTION_REPORT.md`.

| Graph | Nodes | Edges | Max sigma reached | Source |
|:---|---:|---:|---:|:---|
| `nd24k` | 72,000 | 28,715,634 | ~2.4e11 | SuiteSparse ND |
| `roadNet_PA` | 1,090,920 | 3,083,796 | ~1.7e16 | SNAP `roadNet-PA` |
| `youtube` | 1,157,828 | 5,975,248 | — (fp32 precision only) | SNAP `com-Youtube` |

INT_MAX is 2,147,483,647, so `roadNet_PA` overflows it by a factor of about
eight million. To restore them:

```bash
# nd24k -- SuiteSparse, stored as a symmetric matrix with values
wget https://sparse.tamu.edu/MM/ND/nd24k.tar.gz && tar xzf nd24k.tar.gz
python3 suitesparse_sym_to_mtx.py nd24k/nd24k.mtx ../graphData/nd24k.mtx

# roadNet-PA and com-Youtube -- SNAP
wget https://snap.stanford.edu/data/roadNet-PA.txt.gz
wget https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz
python3 snap_to_mtx.py roadNet-PA.txt.gz          ../graphData/roadNet_PA.mtx --undirected
python3 snap_to_mtx.py com-youtube.ungraph.txt.gz ../graphData/youtube.mtx    --undirected
```

`roadNet-PA.txt.gz` already contains both edge directions despite what its
header says; the converter de-duplicates through a set, so `--undirected` is
harmless there.
