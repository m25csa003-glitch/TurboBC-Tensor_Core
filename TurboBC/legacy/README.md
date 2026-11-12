# legacy/

`bcgputc_OLD_geometric_tiling.cu` is the first version of the tensor-core BC
kernel (tcTF32, format 3). It is kept for reference and is **not built**: the
Makefile only compiles `*.cu` in `TurboBC/TurboBC/`, and this file defines the
same functions as the current `../bcgputc.cu`, so the two cannot be linked
together.

## What is different

This version cuts A' along a fixed 16x16 grid (columns 0-15, 16-31, ...) and
stores only the non-empty tiles. The current `bcgputc.cu` uses condensed
tiling instead: for each 16-row window it keeps only the distinct neighbour
columns and cuts them into chunks of 16, so every full chunk has an occupancy
of at least 6.25%.

Batching (16 sources, SpMV -> SpMM) and the three-term TF32 split are the same
in both versions. Only the tile layout changed.

## Why it is kept

It is the "before" in the before/after result reported in
`../REPRODUCTION_REPORT.md` (condensed blocking section) and in the HiPC paper:
on fb_artist a non-empty fixed-grid tile holds 1.12 non-zeros on average
(0.437% occupancy), and this kernel was 4.2x slower than scCSC, while the
condensed-tiling kernel is about 2x faster.

## Running it

To reproduce the old numbers, temporarily replace the current kernel and
rebuild:

```bash
cd TurboBC/TurboBC
cp bcgputc.cu /tmp/bcgputc_current.cu
cp legacy/bcgputc_OLD_geometric_tiling.cu bcgputc.cu
make
./exec ../graphData/fb_artist.mtx 0 0 3 1 32 0 10 0
cp /tmp/bcgputc_current.cu bcgputc.cu && make   # restore
```
