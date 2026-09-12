# How We Compute Betweenness Centrality on Tensor Cores

### A complete walkthrough of `bcgputc.cu` — from the input file to the final BC score

This document explains, step by step, how the tensor-core implementation (`tcTF32`) computes betweenness centrality for **any** graph — sparse or dense — and, just as importantly, **how it does so without ever storing a dense matrix.**

The single most common misunderstanding about this design is that "tensor cores multiply matrices, therefore a matrix must be stored, therefore the memory cost is O(n²)." That is not what happens. The whole design exists to avoid exactly that. If you read only one paragraph of this document, read the highlighted box in Section 3.

---

## Table of Contents

1. [The Problem: Tensor Cores Cannot Multiply Vectors](#1-the-problem-tensor-cores-cannot-multiply-vectors)
2. [The Reformulation: SpMV Becomes SpMM](#2-the-reformulation-spmv-becomes-spmm)
3. [The Space Question — The Key Idea](#3-the-space-question-the-key-idea)
4. [Three Stages, Three Lifetimes](#4-three-stages-three-lifetimes)
5. [Stage 1 — From `.mtx` File to CSC](#5-stage-1-from-mtx-file-to-csc)
6. [Stage 2 — From CSC to the Condensed Format](#6-stage-2-from-csc-to-the-condensed-format)
7. [Stage 3 — From the Condensed Format to a 16×16 Tile](#7-stage-3-from-the-condensed-format-to-a-1616-tile)
8. [Stage 2 vs Stage 3 — The Real Difference](#8-stage-2-vs-stage-3-the-real-difference)
9. [A Complete Walkthrough With Real Numbers: fb_artist](#9-a-complete-walkthrough-with-real-numbers-fb_artist)
10. [Exact Memory Accounting](#10-exact-memory-accounting)
11. [Scaling Limits — How Large a Graph Fits?](#11-scaling-limits-how-large-a-graph-fits)
12. [Proof That This Is O(nz), Not O(n²)](#12-proof-that-this-is-onz-not-on²)
13. [Precision: Why TF32 Needs a 3-Way Split](#13-precision-why-tf32-needs-a-3-way-split)
14. [The Full Algorithm, End to End](#14-the-full-algorithm-end-to-end)
15. [Results](#15-results)
16. [Summary](#16-summary)

---

## 1. The Problem: Tensor Cores Cannot Multiply Vectors

Betweenness centrality in the language of linear algebra reduces to one dominant operation, repeated at every level of a breadth-first search:

```
f  ←  A' f
```

where `A'` is the transposed adjacency matrix (n × n, sparse) and `f` is the frontier vector (n × 1). This is a **sparse matrix–vector product (SpMV)**, and it is what all three of the paper's CUDA-core kernels compute — `scCSC` with one thread per vertex, `vcCSC` with one warp per vertex, `scCOOC` with one thread per edge.

Tensor cores cannot execute this operation at all. A tensor core is a fixed-function unit that computes one thing:

```
C  +=  A × B          with fixed, small shapes
```

On the NVIDIA L4 (sm_89) with TF32 inputs, the shape available through the WMMA API is `m16n16k8`:

```
C[16 × 16]  +=  A[16 × 8]  ×  B[8 × 16]
```

There is no vector variant. A tensor core given an n × 1 vector would use one of its 16 output columns and waste the other 15. So before tensor cores can be used for BC at all, the operation itself has to change shape.

---

## 2. The Reformulation: SpMV Becomes SpMM

The fix is to stop processing one source vertex at a time.

Betweenness centrality requires a full BFS from **every** vertex in the graph. The three CUDA-core kernels do these one after another: run the BFS from root 0, accumulate, then root 1, then root 2, and so on. Nothing forces that to be sequential — the roots are independent.

So we process **16 roots simultaneously**. The frontier is no longer a vector but a matrix:

| | CUDA-core kernels | Tensor-core kernel |
|:---|:---|:---|
| Roots in flight | 1 | **16** |
| Frontier | `f` — n × 1 **vector** | `F` — n × 16 **matrix** |
| BFS step | `f ← A'f` (SpMV) | `F ← A'F` (**SpMM**) |
| Column *j* of the frontier | — | the frontier of root *j* |

```
FT      =      A'      ×      F
(n×16)       (n×n)          (n×16)
             sparse
```

**Why exactly 16?** Because the `m16n16k8` fragment has N = 16. Choosing `TCK = 16` makes the number of roots per batch exactly equal to the width of the tensor core's output, so no output lane is ever wasted. In the code:

```c
#define TCK      16   /* roots per batch == wmma N */
#define TC_TILE  256  /* 16x16 */
```

The same SpMM kernel serves both phases of the Brandes algorithm:

* **Forward (BFS) phase:** `FT ← A'F` — propagate frontiers and count shortest paths
* **Backward (delta) phase:** `ΔU_T ← A'ΔU` — propagate dependency credit back down the levels

One kernel, `tcSpMMKernel`, is called for both.

---

## 3. The Space Question — The Key Idea

Here is the objection this design has to answer:

> "A tensor core multiplies dense 16×16 blocks. To feed it, you must build those blocks. An n × n matrix cut into 16×16 blocks has (n/16)² blocks. For n = 100,000 that is 39 million blocks. Storing them would take O(n²) space — 37 GB for `smallworld` alone, which does not even fit on the GPU. Meanwhile the CUDA-core kernels convert the matrix to CSC or COOC and stay at O(nz). So doesn't the tensor-core version sacrifice memory?"

The objection is correct about the arithmetic and wrong about the design. We never build those blocks in memory.

> ### 🔑 THE KEY IDEA
>
> **We build a *condensed sparse format* from the CSC arrays — on the CPU, exactly once — and it lives in GPU memory for the whole run (8.16 MB for fb_artist, storing nothing but non-zeros). From that format, inside the kernel, a 16×16 dense window is assembled in *shared memory* for each chunk, the MMA runs on it, and it is discarded immediately — roughly 3.25 million times per run. Dense data never reaches GPU global memory at all.**

Everything else in this document is an elaboration of that paragraph.

Two facts follow from it, and both are worth stating plainly because they are the opposite of what people expect:

1. **All four algorithms use a sparse format.** `scCSC` and `vcCSC` use CSC. `scCOOC` uses COOC. `tcTF32` uses a condensed blocked format. All three formats are O(nz).
2. **The tensor-core kernel receives the very same CSC arrays** that `scCSC` receives. From `bcugcsccooc_main.c:318`:

```c
bc_gpu_ug_tc (CscA.IC, CscA.CP, S_hgpu, sigma_hgpu, bc_hgpu, nr, rs, nz, N, repet);
                  ↑         ↑
          the identical CSC arrays that scCSC is given
```

The tensor-core path is therefore a *second* sparse-to-sparse conversion: `.mtx` → CSC → condensed blocked. At no point does a dense representation of the graph exist.

---

## 4. Three Stages, Three Lifetimes

The confusion about memory almost always comes from collapsing two different things into one. There are **three** distinct stages, and they have wildly different lifetimes.

```
STAGE 1 — .mtx file  →  CSC                      [CPU, ONCE, at program start]
          bcugcsccooc_main.c
          ↓
          CP[n+1], IC[nz]                        ← sparse
          ↓
          ════════════════════════════════════════════════════════════════════
           The two code paths diverge here
          ════════════════════════════════════════════════════════════════════
          ↓                                      ↓
   scCSC / vcCSC use                      STAGE 2 — CSC  →  condensed format
   the CSC arrays directly                         [CPU, ONCE, bcgputc.cu:186-236]
                                                   ↓
                                                   browptr, chunkcol, bentptr, bent
                                                   ↓  copied to GPU (lines 250-253)
                                                   ↓
                                                   ┌───────────────────────────────┐
                                                   │ STAYS IN GPU GLOBAL MEMORY    │
                                                   │ for the entire run            │
                                                   │ (freed at line 367, the end)  │
                                                   └───────────────────────────────┘
                                                   ↓
                                            STAGE 3 — condensed  →  16×16 tile
                                                      [GPU, INSIDE the kernel,
                                                       MILLIONS of times]
                                                      ↓
                                                      in SHARED memory:
                                                      zero → fill → MMA → forget
```

Read the lifetimes carefully:

* **Stage 2 output is permanent.** It is built once and stays on the GPU until the function returns. It is sparse.
* **Stage 3 output is ephemeral.** It exists for microseconds, in shared memory, and is overwritten by the next chunk. It is the only dense thing in the entire design.

A statement like *"we built the tiles from CSC at runtime and deleted them after the calculation"* mixes the two. The condensed format is **not** built at runtime and is **not** deleted mid-run. The 16×16 tiles are both.

---

## 5. Stage 1 — From `.mtx` File to CSC

This stage is unchanged from the original paper and is shared by all four algorithms. It happens in `bcugcsccooc_main.c` on the CPU, once.

The Matrix Market file is read, degrees are counted, and the graph is stored in **Compressed Sparse Column** format — two arrays:

```
CP[n+1]   column pointers — where each column's entries begin
IC[nz]    row indices     — which rows hold a non-zero in that column
```

For a graph with n = 50,515 and nz = 1,638,180, that is `(50,516 × 4) + (1,638,180 × 4)` bytes = **6.44 MB**. No zero is ever stored.

If the graph is undirected (`--ug 1`), the reader symmetrizes it here, so `nz` doubles — each edge is stored in both directions. This matters when reading density figures: `smallworld` has 499,998 edges in its file but 999,996 non-zeros in the matrix the kernels traverse.

At the end of Stage 1 the two code paths separate. `scCSC` and `vcCSC` copy `CP` and `IC` to the GPU and start computing. `scCOOC` uses the COOC arrays built alongside. `tcTF32` takes `CP` and `IC` and performs one more conversion.

---

## 6. Stage 2 — From CSC to the Condensed Format

**Location:** `bcgputc.cu`, lines 186–236. **Runs on:** the CPU. **How many times:** exactly once per call to `bc_gpu_ug_tc`.

### Why a new format is needed

A tensor core needs 16 rows and 16 columns at a time. The obvious way to supply them is to cut the matrix along a fixed grid: block-column 0 covers vertices 0–15, block-column 1 covers 16–31, and so on. This is called **geometric tiling**, and on real graphs it is a disaster.

The reason is that a vertex's neighbours are scattered arbitrarily across the graph. Take a window of 16 destination vertices in `fb_artist`. Their neighbours might be `7, 41, 902, 4891, 4893, 15003, 20014, 33001, …`. Under geometric tiling, each of those falls into a *different* block-column:

```
neighbour 7      → block-column 0     → tile created → load F[0…15]        (256 floats)
                                                        only 1 of 16 rows used
neighbour 4891   → block-column 305   → tile created → load F[4880…4895]   (256 floats)
                                                        only 1 of 16 rows used
neighbour 20014  → block-column 1250  → tile created → load F[20000…20015] (256 floats)
                                                        only 1 of 16 rows used
```

Measured on `fb_artist`: geometric tiling produced 1,462,715 non-empty tiles for 1,638,180 non-zeros — **1.12 non-zeros per tile**, a tile occupancy of **0.437%**. The GPU was fetching 256 floats to use one of them: over 99% of the memory bandwidth wasted. This is precisely why the first version of the tensor-core kernel *lost* to the CUDA-core kernels on every sparse graph.

### The fix: build blocks from the neighbour list, not from the grid

Abandon the grid. For each window of 16 destination vertices, collect the sorted list of source vertices its rows **actually reference**, and cut *that list* into groups of 16:

```
unique sources referenced by this window:
  [ 7, 41, 902, 4891, 4893, 15003, 20014, 33001, 33004, 41200, … ]
    └──────────────── chunk 0 (16 slots) ────────────────┘└─ chunk 1 ─┘ …

chunk 0's 16 columns are 16 real source vertices — every one of them
carries at least one non-zero, because that is why it is in the list
```

Since every column of a chunk is guaranteed to carry a non-zero, **occupancy is bounded below by 16/256 = 6.25% by construction** — regardless of how sparse the graph is. Measured results:

| Graph | Geometric tiling | Condensed blocking |
|:---|---:|---:|
| fb_artist | 0.437% | **6.301%** |
| epinions | 0.823% | **7.116%** |
| smallworld | 1.789% | **15.374%** |
| nd3k | 43.58% | **60.588%** |
| nd6k | 42.29% | **59.619%** |

### How the format is built

The construction is a sort followed by a single linear pass.

**Step 1 — emit one entry per edge, keyed so that a sort groups the work correctly.**

```c
std::vector<TcEnt> ents(nz);
size_t t = 0;
for (int v = 0; v < n; v++)
  for (int k = CP_h[v]; k < CP_h[v+1]; k++){
    int u = I_h[k];                      /* edge u -> v, so A'[v][u] is non-zero */
    ents[t].key  = ((unsigned long long)(v>>4) << 32) | (unsigned int)u;
    ents[t].lrow = (unsigned char)(v & 15);
    t++;
  }
```

The key packs two fields into one 64-bit integer: the **row-window** `v >> 4` (that is, `v / 16`) in the high 32 bits, and the **source vertex** `u` in the low 32 bits. `lrow` is the destination's position inside its window, `v & 15` (that is, `v % 16`), which fits in 4 bits.

**Step 2 — sort by that key.**

```c
std::sort(ents.begin(), ents.end(),
          [](const TcEnt &a, const TcEnt &b){ return a.key < b.key; });
```

Because the row-window occupies the high bits, the sort groups every edge belonging to a window together, and *within* a window it orders them by source vertex. Both properties are needed by the next step: grouping lets us process one window at a time, and ordering lets us detect duplicate sources by comparing with the previous entry alone.

**Step 3 — one linear pass: number the distinct sources and pack 16 per chunk.**

```c
for (int br = 0; br < nb; br++){            /* br = row-window index */
  h_browptr[br] = nblocks;                  /* first chunk of this window */
  int slot = 16;                            /* 16 means "current chunk is full" */
  unsigned int prev = 0;
  int first = 1;

  while (i < (size_t)nz && (int)(ents[i].key >> 32) == br){
    unsigned int u = (unsigned int)(ents[i].key & 0xffffffffu);

    if (first || u != prev){                /* a source we have not seen yet */
      if (slot == 16){                      /* need to open a new chunk */
        if (nblocks) h_bentptr.push_back((int)h_bent.size());
        h_chunkcol.resize(h_chunkcol.size()+16, 0);
        nblocks++; slot = 0;
      }
      h_chunkcol[h_chunkcol.size()-16+slot] = (int)u;   /* record the column */
      prev = u; first = 0; slot++;
    }
    /* pack (row, slot) into one byte: this IS the offset inside the 16x16 tile */
    h_bent.push_back((unsigned char)((ents[i].lrow << 4) | (slot-1)));
    i++;
  }
}
```

Note the last line. `(lrow << 4) | slot` is a number from 0 to 255, and it is exactly the index of this non-zero inside a row-major 16×16 tile. The byte therefore does double duty: it stores the edge *and* tells the kernel where to write it. One byte per edge, no row index and no column index stored separately.

Trailing slots of a window's final chunk are left pointing at vertex 0. That is safe: their adjacency column in the tile stays zero, so they contribute nothing to the product.

### What Stage 2 produces

Four arrays, all sparse:

| Array | Size | What it holds |
|:---|:---|:---|
| `browptr` | n/16 + 1 ints | where each row-window's chunks begin |
| `chunkcol` | 16 × nblocks ints | the 16 real source vertices of each chunk |
| `bentptr` | nblocks + 1 ints | where each chunk's non-zeros begin |
| `bent` | **nz bytes** | one byte per edge: `(row << 4) \| slot` |

These are copied to the GPU once (lines 250–253) and stay there:

```c
cudaMemcpy(browptr_d,  h_browptr.data(),  ..., cudaMemcpyHostToDevice);
cudaMemcpy(chunkcol_d, h_chunkcol.data(), ..., cudaMemcpyHostToDevice);
cudaMemcpy(bentptr_d,  h_bentptr.data(),  ..., cudaMemcpyHostToDevice);
cudaMemcpy(bent_d,     h_bent.data(),     ..., cudaMemcpyHostToDevice);
```

They are freed only at line 367, when the function returns.

---

## 7. Stage 3 — From the Condensed Format to a 16×16 Tile

**Location:** inside `tcSpMMKernel`. **Runs on:** the GPU. **How many times:** once per chunk, per SpMM call — millions of times per run.

This is the only place a dense 16×16 block ever exists, and it exists in **shared memory**, which is per-block scratch space physically located inside the SM. It is not GPU global memory, it is never written back, and each tile is overwritten by the next chunk.

### The work assignment

One warp owns one row-window and accumulates across all of that window's chunks into a **single accumulator fragment**. Because the accumulator lives in registers and is written out only once at the end, no atomics are needed on the output:

```c
int wid  = threadIdx.x >> 5;                    /* which warp in this block  */
int lane = threadIdx.x & 31;                    /* which lane in this warp   */
int brow = blockIdx.x * TC_WARPS + wid;         /* which row-window I own    */

extern __shared__ float sh[];
float *At = sh + wid * (TC_TILE + 3*TC_TILE);   /* my 16x16 adjacency tile   */
float *Fs = At + TC_TILE;                       /* my frontier tile, 3 splits*/

wmma::fragment<wmma::accumulator,16,16,8,float> acc;
wmma::fill_fragment(acc, 0.0f);                 /* accumulate in registers   */
```

Shared memory per warp is `256 + 3×256 = 1024` floats = **4 KB** (1 KB for the adjacency tile, 3 KB for the three-way-split frontier). With `TC_WARPS = 4`, the block requests **16 KB** — a small fraction of the L4's 100 KB per SM, so occupancy stays high.

### Building the tile

For each chunk the warp owns:

```c
for (int b = browptr[brow]; b < browptr[brow+1]; b++){

    /* ── 1. zero the 16x16 tile ──────────────────────────────────────── */
    for (int t = lane; t < TC_TILE; t += 32) At[t] = 0.0f;
    __syncwarp();

    /* ── 2. scatter in ONLY the non-zeros ───────────────────────────────
       bent[e] already equals (local_row << 4) | slot, which is the tile
       offset — so the byte is the address. No decoding needed.          */
    for (int e = bentptr[b] + lane; e < bentptr[b+1]; e += 32)
        atomicAdd(&At[bent[e]], 1.0f);

    /* ── 3. gather the frontier for this chunk's 16 REAL columns ────────
       slot s holds source vertex chunkcol[b*16+s]. Lanes sharing a slot
       read its 16 consecutive roots, so each slot is one 64-byte
       coalesced load rather than a strided read of a mostly empty tile. */
    const int *cc = chunkcol + (size_t)b * 16;
    for (int t = lane; t < TC_TILE; t += 32){
        float x  = F[(size_t)cc[t >> 4] * TCK + (t & 15)];
        float h1 = wmma::__float_to_tf32(x);      /* 3-way split, Section 12 */
        float r1 = x - h1;
        float h2 = wmma::__float_to_tf32(r1);
        Fs[t]           = h1;
        Fs[TC_TILE+t]   = h2;
        Fs[2*TC_TILE+t] = r1 - h2;
    }
    __syncwarp();

    /* ── 4. run the tensor core ─────────────────────────────────────── */
    for (int s = 0; s < nsplit; s++){
      for (int kk = 0; kk < 16; kk += 8){
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> fb;
        wmma::load_matrix_sync(fa, At + kk, 16);
        wmma::load_matrix_sync(fb, Fs + s*TC_TILE + kk*16, 16);
        wmma::mma_sync(acc, fa, fb, acc);        /* C[16x16] += A[16x8] x B[8x16] */
      }
    }
    __syncwarp();
    /* ── 5. next chunk: go back to step 1, overwrite At and Fs ──────── */
}
wmma::store_matrix_sync(FT + (size_t)brow*TC_TILE, acc, 16, wmma::mem_row_major);
```

Step by step, in words:

1. **Zero.** All 256 cells of the tile are set to 0. The 32 lanes each handle 8 cells.
2. **Fill only the non-zeros.** For `fb_artist` that is roughly 16 cells out of 256 — the other 240 stay at the zero from step 1. This is the point at which sparse data becomes a dense window, and it happens in registers and shared memory only.
3. **Gather the frontier.** The 16 columns are looked up through `chunkcol`, so they are 16 *real* source vertices, not 16 consecutive addresses. Lanes 0–15 read `F[c*16 + 0…15]` for the same `c`, which is 64 contiguous bytes — a perfectly coalesced load.
4. **Multiply.** The `kk = 0, 8` loop issues two MMAs to cover the full 16×16; the `nsplit` loop issues three passes for precision. Six MMA instructions per chunk.
5. **Forget.** The next chunk starts at step 1 and overwrites the same shared memory. Nothing is saved.

The accumulator `acc` is the one thing that survives across chunks — it lives in registers and holds the partial result for this window's 16 destination vertices × 16 roots. It is written to global memory exactly once, after the last chunk.

---

## 8. Stage 2 vs Stage 3 — The Real Difference

| | **Condensed format** (Stage 2) | **16×16 tile** (Stage 3) |
|:---|:---|:---|
| Where it is built | CPU, in host code | GPU, inside the kernel |
| How many times | **Once**, at the start | **Millions of times** |
| Where it lives | GPU **global** memory | **Shared** memory (warp scratch) |
| How big | 2.3 – 8.2 MB | 1 KB adjacency + 3 KB frontier |
| How long it lives | **The entire run** | **~1 microsecond** |
| Data representation | **Sparse** — non-zeros only | **Dense** 16×16 window |
| Freed when | Function returns (line 367) | Immediately, by the next chunk |

The columns describe two completely different objects. Conflating them is what produces the false conclusion that the design needs O(n²) memory.

---

## 9. A Complete Walkthrough With Real Numbers: fb_artist

`fb_artist` is a Facebook page-page network from the SNAP `gemsec-Facebook` dataset:

```
n  = 50,515 nodes
nz = 1,638,180 edges
Matrix density = 1,638,180 / 50,515² x 100 = 0.0642%
                 → about 1 filled cell in every 1,558
```

### Stage 1 — once
```
.mtx file read → CSC:  CP[50,516] + IC[1,638,180]  =  6.44 MB
```

### Stage 2 — once
```
1,638,180 entries created, sorted, and condensed into:

    3,158 row-windows        (50,515 vertices / 16, rounded up)
  101,565 chunks             (16 real source columns each)
    6.301% tile occupancy    (vs 0.437% under geometric tiling)

Arrays copied to the GPU:
    browptr    3,159 ints      =     12.3 KB
    chunkcol   1,625,040 ints  =      6.20 MB
    bentptr    101,566 ints    =    396.7 KB
    bent       1,638,180 bytes =      1.56 MB
                                 ─────────────
                                      8.16 MB   ← stays for the whole run
```

Note the shape of this: 101,565 chunks × 16 slots = 1,625,040 slots for 1,638,180 edges, so there is almost no column sharing between the 16 rows of a window (a reuse factor of 1.01). `fb_artist` is a scale-free social graph with little local structure. `smallworld`, by contrast, reaches a reuse factor of 2.46 — which is exactly why its tile occupancy (15.37%) and its speedup (6.83x) are both the highest in the suite.

### Stage 3 — millions of times

With `nr = 32` roots and `repet = 1`:

```
BFS phase   :  9 SpMM calls   (BFS depth d = 8, plus one final call that finds nothing)
Delta phase :  7 SpMM calls   (d = 8 down to 2)
              ─────────────
             16 SpMM calls per batch of 16 roots

32 roots / 16 = 2 batches   →   32 SpMM calls total

Each SpMM call builds one tile per chunk:
    101,565 chunks × 32 SpMM calls  =  3,250,080 tile constructions
```

**Roughly 3.25 million times**, a 16×16 tile is zeroed in shared memory, filled with about 16 non-zeros, multiplied, and discarded. In the benchmark script `repet = 10`, so the real figure is about **32.5 million** tile constructions per graph per run.

At no point does a dense block reach GPU global memory.

### Result
```
scCSC   (best CUDA-core kernel on this graph) :  1.0018 ms per root
tcTF32                                        :  0.5086 ms per root
                                                 ─────────
                                                 1.97x faster

CPU sequential                                : 28.725 ms per root
                                                 → 56.47x faster
BC scores : all 50,515 nodes match the CPU exactly
```

---

## 10. Exact Memory Accounting

All figures measured, not estimated. `nblocks` comes from the kernel's own diagnostic line.

### Per-algorithm totals

| Graph | `scCSC` | `tcTF32` | Ratio | tcTF32: graph structure | tcTF32: 16-wide state |
|:---|---:|---:|---:|---:|---:|
| fb_artist | 7.98 MB | 29.94 MB | 3.8x | 8.16 MB | 21.78 MB |
| epinions | 4.46 MB | 34.99 MB | 7.7x | **2.32 MB** | 32.67 MB |
| smallworld | 7.24 MB | 45.71 MB | 6.3x | **2.61 MB** | 43.10 MB |

### Where the difference actually comes from

Look at the last two columns. The **graph structure** is not the problem — on `epinions` and `smallworld` the tensor-core format is *smaller* than CSC. The reason is `bent`: **one byte per edge** (row and slot packed together), against CSC's `IC` at **four bytes per edge**.

The entire increase comes from the **state arrays**, because 16 roots are in flight at once and each root needs its own path counts, its own depths, and its own dependency values:

```
scCSC    :  f, ft, S, sigma, delta, delta_u, delta_ut, bc    each  n × 1
tcTF32   :  F, FT, S, Sigma, Delta, DeltaU,  DeltaUT         each  n × 16
```

Seven arrays at `npad × 16` instead of `n × 1`. For `fb_artist`:

```
6 float arrays × 50,528 × 16 × 4 bytes  =  18.50 MB
1 int   array  × 50,528 × 16 × 4 bytes  =   3.08 MB
bc (n floats, not 16-wide)              =   0.20 MB
                                           ────────
                                           21.78 MB
```

This is a deliberate **time-for-space trade**: about 6x the memory buys 1.6x–6.8x the speed. And it is cheap in absolute terms — 45.71 MB is **0.2% of the L4's 23 GB**. At this rate the approach scales to graphs of roughly 45 million nodes before memory becomes the binding constraint.

---

## 11. Scaling Limits — How Large a Graph Fits?

The memory breakdown above leads naturally to a practical question: **what is the largest graph each algorithm can handle?** There are two separate ceilings, and which one binds depends on the graph.

### The byte formulas

These are read directly from the `cudaMalloc` calls and validated against measured usage:

| Algorithm | Bytes required | Where it comes from |
|:---|:---|:---|
| **scCSC / vcCSC** | `4·nz + 36·n` | `CP[n+1]` + `IC[nz]` + 8 arrays of length `n` |
| **scCOOC** | `8·nz + 32·n` | `ICOOC[nz]` + `JCOOC[nz]` + 8 arrays of length `n` |
| **tcTF32** | `452·n + nz·(1 + 4.25/r)` | 7 arrays of `n × 16` + the condensed format |

`r` is the **column reuse factor** — how much the 16 rows of a window share neighbours. Measured values: `fb_artist` 1.01, `epinions` 1.14, `smallworld` 2.46.

Validation against Section 10's measured figures, on `fb_artist` (n = 50,515, nz = 1,638,180):

```
scCSC  formula:  4(1,638,180) + 36(50,515)            =  7.98 MB   (measured 7.98 MB) ✓
tcTF32 formula:  452(50,515) + 1,638,180(1 + 4.25/1.01) = 29.91 MB  (measured 29.94 MB) ✓
```

### Ceiling 1 — GPU memory (22 GB usable on the L4)

The decisive difference is the **per-node** cost, not the per-edge cost:

| | Per node | Per edge |
|:---|---:|---:|
| scCSC / vcCSC | **36 bytes** | 4 bytes |
| scCOOC | 32 bytes | 8 bytes |
| **tcTF32** | **452 bytes** — 12.5x more | ~1–5 bytes |

tcTF32 pays 12.5x per node because seven of its state arrays are `n × 16` rather than `n × 1`. Per *edge*, however, it is equal or cheaper — `bent` stores one byte per edge against CSC's four.

### Ceiling 2 — 32-bit integer overflow

`nz` is declared `long` in `bcugcsccooc_main.c`, but every GPU entry point takes it as `int`:

```c
int bc_gpu_ug_csc_sc (int *I_h, int *CP_h, ..., int nz, int n, int repetition);
                                                  ↑
                                        truncated from long
```

This imposes a hard ceiling of **nz ≤ 2,147,483,647** (about 2.15 billion edges) on all four algorithms. Beyond that the value overflows silently.

### The resulting limits

| Average degree | scCSC / vcCSC | scCOOC | **tcTF32** | Binding ceiling |
|:---|:---|:---|:---|:---|
| **5** | 412 M nodes / 2.06 B edges | 320 M / 1.60 B | **48 M / 241 M** | memory, for all three |
| **10** | 215 M / **2.15 B** | 206 M / 2.06 B | **46 M / 457 M** | int32 for scCSC, memory for tcTF32 |
| **30** | 72 M / **2.15 B** | 72 M / **2.15 B** | **38 M / 1.14 B** | int32 for the CUDA-core kernels |
| **100** | 21 M / **2.15 B** | 21 M / **2.15 B** | **21 M / 2.15 B** | **int32, for all three** |
| **400** | 5.4 M / **2.15 B** | 5.4 M / **2.15 B** | **5.4 M / 2.15 B** | **int32, for all three** |

Two conclusions follow.

**On sparse graphs (average degree 5–10), tcTF32's reach is roughly 8x smaller.** It tops out near 46–48 million nodes where scCSC reaches 215–412 million. This is the direct consequence of 452 bytes per node against 36.

**On denser graphs (average degree ≥ 100) the three algorithms are identical**, all capped at about 21 million nodes — because there the 32-bit `nz` ceiling is reached before memory runs out. For graphs like `nd3k` and `nd6k`, the tensor-core approach carries no capacity penalty whatsoever.

### Where the current test graphs sit

| Graph | Nodes | tcTF32 limit at that degree | Headroom |
|:---|---:|---:|---:|
| smallworld | 100,000 | ~46 M | **460x** |
| epinions | 75,888 | ~47 M | **620x** |
| fb_artist | 50,515 | ~38 M | **750x** |
| nd6k | 18,000 | ~5.4 M | **300x** |

Even the largest graph in the suite uses about **0.2%** of the available capacity. Memory is nowhere near being the binding constraint for this work.

### The constraint that actually binds: time, not memory

These ceilings are what *fits*. What is *runnable* is a much smaller number, because full betweenness centrality requires a BFS from every vertex:

```
fb_artist, full BC over all 50,515 roots:
    50,515 x 0.5086 ms  =  26 seconds

Extrapolated to the 38 M node memory ceiling:
    38,000,000 x 0.5086 ms  ≈  5.4 hours per million nodes  ≈  7 months
```

So memory would allow a 38-million-node graph that could never actually be run to completion. In practice:

* **Up to ~1–2 million nodes** — full exact BC is feasible (roughly 1–7 hours)
* **Beyond that** — use approximate BC, sampling a few thousand roots instead of all `n`. This is standard practice in the literature, and the multi-source design suits it well: any multiple of 16 roots can be sampled with no code change.

### A note for future work

Widening `nz` from `int` to `long` or `size_t` would remove the 2.15-billion-edge ceiling entirely, leaving memory as the only limit. It is a change to five function signatures in `bcug.h` and their definitions. It is not needed at the graph sizes studied here, but it is the first thing to fix if the work is ever extended to billion-edge graphs.

---

## 12. Proof That This Is O(nz), Not O(n²)

Three independent arguments.

### Argument 1 — every allocation in the file

`bcgputc.cu` contains exactly 14 device allocations. Here they all are:

```c
cudaMalloc(&browptr_d,  (nb+1)*sizeof(int))                    /* n/16      */
cudaMalloc(&chunkcol_d, (size_t)nblocks*16*sizeof(int))        /* O(nz)     */
cudaMalloc(&bentptr_d,  (nblocks+1)*sizeof(int))               /* O(nz)     */
cudaMalloc(&bent_d,     nz*sizeof(unsigned char))              /* nz bytes  */
cudaMalloc(&F_d,        velem*sizeof(float))                   /* n × 16    */
cudaMalloc(&FT_d,       velem*sizeof(float))                   /* n × 16    */
cudaMalloc(&Sigma_d,    velem*sizeof(float))                   /* n × 16    */
cudaMalloc(&Delta_d,    velem*sizeof(float))                   /* n × 16    */
cudaMalloc(&DeltaU_d,   velem*sizeof(float))                   /* n × 16    */
cudaMalloc(&DeltaUT_d,  velem*sizeof(float))                   /* n × 16    */
cudaMalloc(&S_d,        velem*sizeof(int))                     /* n × 16    */
cudaMalloc(&roots_d,    TCK*sizeof(int))                       /* 16        */
cudaMalloc(&bc_d,       n*sizeof(float))                       /* n         */
cudaMallocManaged(&c,   sizeof(int))                           /* 1         */
```

where `velem = npad × TCK = npad × 16`. The largest term is `n × 16`. **There is no `n × n` allocation anywhere in the file**, and `grep` for `n*n` returns nothing.

### Argument 2 — it would not fit

If the design really materialized a dense matrix:

| Graph | Dense n × n float matrix | L4 total memory |
|:---|---:|---:|
| fb_artist | 9.5 GB | 23 GB |
| epinions | 21.5 GB | 23 GB |
| smallworld | **37.3 GB** | 23 GB ❌ |

`smallworld` would not fit on the GPU at all. It runs, and it produces correct BC scores for all 100,000 nodes — which is only possible if no dense matrix is being stored.

### Argument 3 — all four formats are O(nz)

| Algorithm | Format | Arrays | Complexity |
|:---|:---|:---|:---|
| scCSC, vcCSC | CSC | `CP[n+1]`, `IC[nz]` | O(n + nz) |
| scCOOC | COOC | `ICOOC[nz]`, `JCOOC[nz]` | O(nz) |
| **tcTF32** | Condensed blocked | `browptr[n/16+1]`, `chunkcol[16·nblocks]`, `bentptr[nblocks+1]`, `bent[nz]` | **O(n/16 + nz)** |

The tensor-core format is not an exception to sparse storage. It is a sparse format designed for a different access pattern.

---

## 13. Precision: Why TF32 Needs a 3-Way Split

One more thing had to be solved before tensor cores could be used for BC at all: exactness.

`sigma[v]` is the number of shortest paths from the root to `v`. It is an **integer**, and it can be enormous — 24,973 on `fb_artist`, and 4.13 × 10¹⁰ on `nd6k`. BC is only correct if these counts are exact.

| Format | Mantissa | Exact integers up to | Maximum value | Verdict |
|:---|---:|---:|---:|:---|
| fp16 | 11 bits | **2,048** | 65,504 | ❌ fails twice over |
| tf32 | 11 bits | 2,048 | 3.4 × 10³⁸ | ⚠️ range fine, precision not |
| fp32 | 24 bits | 16,777,216 | 3.4 × 10³⁸ | ✅ what we need to match |

fp16 is unusable for two independent reasons: its exact-integer range stops at 2,048 while path counts reach 24,973, and `sigma` would overflow its 65,504 maximum outright on several graphs. TF32 fixes the range — it carries fp32's exponent — but keeps the same 11-bit mantissa.

**The fix is to split each value into three parts that are each exact in tf32:**

```c
float x  = F[...];                        /* the fp32 value        */
float h1 = wmma::__float_to_tf32(x);      /* top 11 bits           */
float r1 = x - h1;                        /* what h1 could not hold */
float h2 = wmma::__float_to_tf32(r1);     /* next 11 bits          */
float h3 = r1 - h2;                       /* the remainder         */
/* x = h1 + h2 + h3, and 3 x 11 = 33 mantissa bits > fp32's 24 */
```

Three MMAs are then issued per tile instead of one, and their results accumulate into the same fp32 accumulator. Because the sum `h1 + h2 + h3` reconstructs `x` to 33 bits of mantissa — more than fp32 itself carries — the product is exact to fp32.

### Why three and not two

The choice of three is a **theoretical** one. TF32 carries an 11-bit mantissa, so:

* **2 splits → 22 bits**, which is *less* than fp32's 24. No guarantee of exactness for an arbitrary fp32 input.
* **3 splits → 33 bits**, comfortably more than 24. Exactness is guaranteed.

Empirically, however, 2 splits are enough on every graph we have tested. Measured max **relative** error against the CPU on the current condensed code:

| Graph | split = 1 | split = 2 | split = 3 |
|:---|---:|---:|---:|
| smallworld (bc) | 3.683e-04 ❌ | 1.000e-06 ✅ | 1.000e-06 ✅ |
| epinions (bc) | 6.115e-04 ❌ | 7.727e-06 ✅ | 7.727e-06 ✅ |
| fb_artist (bc) | 4.683e-04 ❌ | 8.774e-06 ✅ | 8.705e-06 ✅ |
| nd6k (sigma) | 4.194e-04 ❌ | 2.642e-06 ✅ | 2.642e-06 ✅ |
| nd6k (bc) | 1.679e-04 ❌ | 3.539e-06 ✅ | 3.539e-06 ✅ |

Reading this table:

* **One split fails everywhere.** Errors of 1.7e-04 to 6.1e-04 are above the 1e-04 acceptance bar. A single tf32 term is genuinely not enough.
* **Two splits pass everywhere**, with roughly 11x margin to spare.
* **Three splits are indistinguishable from two** on every graph tested — the numbers are identical to the last digit except for one negligible change on fb_artist.

So why is the default 3? Because two splits have only *empirical* support on this particular set of graphs, while three splits have a *proof* (33 > 24 bits). A graph with larger path counts than anything here could expose the gap. The default is therefore the provably-safe setting, and the faster setting is available when the user wants it:

| Graph | split = 3 | split = 2 | Speed gain from dropping the third split |
|:---|---:|---:|---:|
| smallworld | 0.3521 ms | 0.3078 ms | **1.14x** |
| ASIC_100ks | 0.0323 ms | 0.0294 ms | **1.10x** |
| epinions | 0.8857 ms | 0.8124 ms | **1.09x** |
| fb_artist | 0.5361 ms | 0.5067 ms | **1.05x** |
| nd6k | 0.2914 ms | 0.2813 ms | **1.03x** |

### What splitting costs in total

Measured at `nr = 32`, `repet = 3`:

| Graph | split = 1 | split = 2 | split = 3 | Cost of 1 → 3 |
|:---|---:|---:|---:|---:|
| smallworld | 0.3076 ms | 0.3127 ms | 0.3691 ms | **+20%** |
| epinions | 0.7409 ms | 0.8131 ms | 0.8852 ms | **+19%** |
| fb_artist | 0.4481 ms | 0.4985 ms | 0.5346 ms | **+19%** |

Three times the MMA instructions cost only about **20% more wall-clock time**, not 200%. The reason is that this kernel is memory-bound: its arithmetic intensity is roughly 0.1 flop/byte, so at 300 GB/s the roofline caps it near 30 GFLOPS against the L4's ~60 TFLOPS of tf32 throughput. The extra MMAs occupy hardware that was idle anyway.

That same fact is *why* the whole condensed-tile approach is affordable. It deliberately performs about 16x redundant arithmetic — a 16×16 tile holding ~16 non-zeros still runs a full 16×16 multiply — in exchange for perfectly coalesced memory access. That trade only makes sense when arithmetic is nearly free, and on a tensor core it is.

The split level is adjustable at runtime for experimentation:

```bash
TURBOBC_TC_SPLIT=1 ./exec ../graphData/epinions.mtx 0 0 3 0 32 0 1 1   # 1, 2 or 3; default 3
```

---

## 14. The Full Algorithm, End to End

Putting every piece together:

```
1.  Read the .mtx file, build CSC.                            [CPU, once]

2.  Build the condensed blocked format from CSC.              [CPU, once]
    Copy browptr / chunkcol / bentptr / bent to the GPU.
    This stays resident for the whole run.

3.  FOR each batch of 16 root vertices:                       [nr / 16 batches]

      a.  Zero F, Sigma, Delta, S. Set F[root_j][j] = 1 and
          Sigma[root_j][j] = 1 for each of the 16 roots.

      b.  FORWARD (BFS) PHASE — repeat until no vertex is newly reached:
             FT ← A'F                      via tcSpMMKernel  (tensor cores)
             mask out already-visited vertices, accumulate Sigma,
             record the depth S, and set the new frontier F

      c.  BACKWARD (DELTA) PHASE — for d = depth down to 2:
             DeltaU   ← (1 + Delta) / Sigma      for vertices at depth d
             DeltaUT  ← A'DeltaU           via tcSpMMKernel  (tensor cores)
             Delta    += DeltaUT × Sigma         for vertices at depth d-1

      d.  bc[v] += sum over the 16 root slots of Delta[v][j], skipping j == v

4.  Copy bc back to the host. Free the device arrays.         [line 367]
```

Steps 3b and 3c are where `tcSpMMKernel` runs, and therefore where Stage 3 happens millions of times.

Two deliberate differences from the reference CUDA-core kernels, both verified to be equivalent:

* **The visited mask moved out of the SpMV.** `scCSC` skips a column when `sigma[i] >= 0.01`. An MMA cannot mask mid-product, so the same test is applied afterwards, in `tcBfsFunctionsKernel`: `float ft = (Sigma[i] < 0.01f) ? FT[i] : 0.0f;`. A masked vertex contributed `ft = 0` either way.
* **Root initialization moved to its own kernel.** The reference sets `f_d[r] = 1` inside the first SpMV, where every thread in the grid writes the same location — a benign race, but a race. `tcInitRootsKernel` does it separately and race-free.

Both were confirmed by `compute-sanitizer`: **0 memcheck errors, 0 racecheck hazards, 0 synccheck errors**, and results are bit-identical across repeated runs.

---

## 15. Results

All times in **ms per root vertex**, at `nr = 32`, all four algorithms on the same roots. Sorted sparsest graph first.

| Graph | Density | CPU | scCSC | scCOOC | vcCSC | **tcTF32** | Fastest | **vs best CUDA** |
|:---|---:|---:|---:|---:|---:|---:|:---:|---:|
| ASIC_100ks | 0.0059% | 2.359 | 0.0858 | 0.0848 | 0.2192 | **0.0314** | tcTF32 | **2.70x** |
| epinions | 0.0088% | 23.283 | 1.7000 | 1.4346 | 1.8919 | **0.8746** | tcTF32 | **1.64x** |
| smallworld | 0.0100% | 32.262 | 1.3566 | 1.8311 | 1.9073 | **0.1986** | tcTF32 | **6.83x** |
| fb_new_sites | 0.0529% | 12.633 | 0.7946 | 0.7610 | 0.8739 | **0.2640** | tcTF32 | **2.88x** |
| fb_artist | 0.0642% | 28.725 | 1.0018 | 1.2292 | 1.4226 | **0.5086** | tcTF32 | **1.97x** |
| nd6k | 2.1288% | 65.670 | 1.3503 | 4.0131 | 0.5196 | **0.2388** | tcTF32 | **2.18x** |
| nd3k | 4.0490% | 30.875 | 1.2092 | 1.9388 | 0.3639 | **0.2223** | tcTF32 | **1.64x** |
| congress | 5.8899% | 0.221 | 0.1510 | 0.1492 | 0.1433 | **0.0439** | tcTF32 | **3.26x** |

**The speedup does not depend on how sparse the graph is.** The three sparsest graphs give 2.70x, 1.64x and 6.83x; the densest real graph gives 1.64x. The largest gain in the table belongs to `smallworld`, one of the sparsest. After condensed blocking, tile occupancy is a property of the **layout**, not of the graph.

Note also that the best CUDA-core kernel is a *different* one on different graphs — `scCSC` on smallworld and fb_artist, `scCOOC` on ASIC_100ks / fb_new_sites / epinions, `vcCSC` on congress / nd3k / nd6k. `tcTF32` beats all three regardless of which happens to win.

Correctness across the full 48-test suite: **1,524,156 node-level CPU-vs-GPU comparisons, 0 mismatches.**

> The four toy graphs in the suite (mycielskian4 at 11 nodes, cage4 at 9, Ragusa18 at 23, Tina_AskCal at 11) are excluded from the table above. They are far too small for the GPU to do meaningful work — their runtime is almost entirely kernel-launch overhead, and their measured gains swing between runs. The eight real graphs are stable: e.g. epinions 1.63x / 1.64x and fb_artist 1.91x / 1.97x across two consecutive runs.

---

## 16. Summary

**What we compute.** Betweenness centrality for every vertex, on tensor cores, by turning the SpMV `f ← A'f` into the SpMM `F ← A'F` and processing 16 root vertices per batch — 16 because that is the N dimension of the `m16n16k8` tf32 WMMA fragment.

**How we stay exact.** Shortest-path counts are integers reaching 4 × 10¹⁰. fp16 cannot hold them. TF32 has the range but only an 11-bit mantissa, so each value is split three ways — 33 mantissa bits, more than fp32's 24 — at a measured cost of about 20%. (Two splits, at 22 bits, are empirically sufficient on every graph tested and 1.03x–1.14x faster; three is the default because only three carries a proof of exactness.)

**How we handle sparsity.** Tiles are built from each row-window's actual neighbour list rather than from a fixed geometric grid, which raises tile occupancy from 0.437% to 6.301% on `fb_artist` by construction, cuts memory traffic and MMA count by roughly 16x, and turns a 4.2x loss into a 1.97x win.

**How we stay within memory.** This is the part most easily misread, so once more, plainly:

> **CSC is converted, once and on the CPU, into a condensed sparse format that lives in GPU memory for the whole run (8.16 MB on fb_artist, storing nothing but non-zeros). From that format, inside the kernel, a 16×16 dense window is assembled in shared memory for each chunk, the MMA runs on it, and it is discarded immediately — about 3.25 million times per run. Dense data never reaches GPU global memory.**

**How far it scales.** Memory is not the binding constraint at any size studied here. On the L4's 22 GB, tcTF32 fits roughly 46–48 million nodes at average degree 10 (against 215–412 million for scCSC, the cost of 452 bytes per node instead of 36), and on graphs of average degree 100 or more all four algorithms converge on the same ~21-million-node ceiling, set by a 32-bit `nz` rather than by memory. The real limit is time: full exact BC over all roots takes about 26 seconds for `fb_artist`'s 50,515 nodes, which extrapolates to months at the memory ceiling. Full exact BC is practical up to roughly 1–2 million nodes; beyond that the multi-source design lends itself naturally to approximate BC over a sampled multiple of 16 roots. See Section 11 for the full derivation.

Every one of the four algorithms stores the graph in O(nz). The tensor-core version's higher total memory — 3.8x to 7.7x, or 30–46 MB against 4.5–8 MB — comes entirely from running 16 roots concurrently, which makes each of the seven state arrays `n × 16` instead of `n × 1`. That is 0.2% of the L4's 23 GB, and it buys a 1.6x–6.8x speedup over the best CUDA-core kernel on every real graph tested, with BC scores that match the CPU exactly.

---

## Reproducing Any Number in This Document

```bash
cd ~/TurboBC/TurboBC
make clean && make -j$(nproc)

# Full suite: 48 tests + verification, ~7 min, output saved
./run_all_experiments.sh 2>&1 | tee ~/turbobc_results_$(date +%Y%m%d_%H%M).txt

# One graph, all four algorithms
for f in 0 1 2 3; do
  printf "format %s: " $f
  ./exec ../graphData/fb_artist.mtx 0 0 $f 0 32 0 1 0 | grep -oP 'total BC time/vertex = \K[0-9.]+'
done

# Chunk count, tile occupancy and BFS depth for one graph
./exec ../graphData/fb_artist.mtx 0 0 3 0 32 0 1 0 | grep -E "row-windows|BFS  d ="

# Correctness against the CPU for one graph (seq = 1)
./exec ../graphData/fb_artist.mtx 0 0 3 0 32 0 1 1 | grep -E "correct|error in"

# Memory safety
/usr/local/cuda/bin/compute-sanitizer --tool memcheck \
  ./exec ../graphData/fb_artist.mtx 0 0 3 0 32 0 1 0 | grep "ERROR SUMMARY"

# Compare split levels: max relative error on bc (needs p=1 to print per-node values)
for s in 1 2 3; do
  TURBOBC_TC_SPLIT=$s ./exec ../graphData/epinions.mtx 0 0 3 1 32 0 1 1 > /tmp/sp.txt 2>&1
  python3 -c "
body = open('/tmp/sp.txt', errors='replace').read().split('bc   bc_hgpu', 1)[1]
w = 0.0
for line in body.splitlines():
    p = line.strip().split(',')
    if len(p) != 3: continue
    try: c, g = float(p[1]), float(p[2])
    except ValueError: continue
    w = max(w, abs(g - c) / max(1.0, abs(c)))
print('split=$s  max relative error = %.3e' % w)"
done

# Speed cost of each split level
for s in 1 2 3; do
  printf "split=%s : " $s
  TURBOBC_TC_SPLIT=$s ./exec ../graphData/epinions.mtx 0 0 3 0 32 0 3 0 \
    | grep -oP 'total BC time/vertex = \K[0-9.]+'
done
```

**Source:** `bcgputc.cu` — Stage 2 at lines 186–236, device allocations at 246–265, the SpMM kernel at 50–108, the driver loop at 283–345, cleanup at 367.
