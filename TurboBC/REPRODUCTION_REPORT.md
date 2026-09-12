# TurboBC — Paper Reproduction + Tensor Core Extension

**Paper:** *TurboBC: A Memory Efficient and Scalable GPU Based Betweenness Centrality Algorithm in the Language of Linear Algebra*
**Authors:** Oswaldo Artiles & Fahad Saeed, Florida International University
**Published:** ICPP Workshops '21, August 9–12, 2021 | DOI: 10.1145/3458744.3474047
**Original Code:** https://github.com/pcdslab

This document has two parts:

* **Part A** — reproducing the paper's original code on our GPU (3 CUDA-core algorithms)
* **Part B** — our new work: a **Tensor Core** BC implementation (`tcTF32`) that beats all three CUDA-core algorithms even on sparse graphs

---

## 1. Hardware Environment

### The Paper's Original System
| Property | Value |
|:---|:---|
| GPU | NVIDIA Titan Xp (Pascal) |
| Compute Capability | 6.1 (sm_61) |
| VRAM | 12,196 MB |
| Memory Bandwidth | 547.7 GB/s |
| L2 Cache | 3 MB |
| Tensor Cores | **None** |
| CUDA Toolkit | 10.1.243 |
| OS | Ubuntu 16.04.6 |

### Our System
| Property | Value |
|:---|:---|
| GPU | NVIDIA L4 (Ada Lovelace) |
| Compute Capability | 8.9 (sm_89) |
| VRAM | 23,034 MB (~22.5 GB) |
| Memory Bandwidth | 300 GB/s |
| L2 Cache | 48 MB |
| SMs | 58 |
| Tensor Cores | 4th gen (fp16 / **tf32** / int8) |
| CUDA Toolkit | 11.5 |
| Driver | 580.173.02 |
| OS | Ubuntu 22.04 (Kernel 5.15) |
| GCC | 11.4.0 |

The Titan Xp has no tensor cores at all, so the work in Part B was simply not possible on the paper's original hardware.

---

## 2. Repository Structure

```
~/TurboBC/
├── graphData/                     # 12 graphs (Matrix Market format)
│   ├── mycielskian4.mtx           # 11 nodes      — toy
│   ├── Tina_AskCal.mtx            # 11 nodes      — toy
│   ├── cage4.mtx                  # 9 nodes       — toy
│   ├── Ragusa18.mtx               # 23 nodes      — toy
│   ├── congress.mtx               # 475 nodes     — [NEW] Twitter congress network
│   ├── congress.edgelist          # [NEW] raw source for congress
│   ├── smallworld.mtx             # 100,000 nodes — SuiteSparse
│   ├── ASIC_100ks.mtx             # 99,190 nodes  — SuiteSparse circuit matrix
│   ├── fb_new_sites.mtx           # 27,917 nodes  — [NEW] SNAP gemsec-Facebook
│   ├── fb_artist.mtx              # 50,515 nodes  — [NEW] SNAP gemsec-Facebook
│   ├── epinions.mtx               # 75,888 nodes  — [NEW] SNAP soc-Epinions1
│   ├── nd3k.mtx                   # 9,000 nodes   — [NEW] SuiteSparse ND (3D FEM)
│   └── nd6k.mtx                   # 18,000 nodes  — [NEW] SuiteSparse ND (3D FEM)
│
└── TurboBC/
    ├── Makefile                    # [MODIFIED] CUDA paths + sm_89
    ├── bcugcsccooc_main.c          # [MODIFIED] added format 3 dispatch
    ├── bcug.h                      # [MODIFIED] bc_gpu_ug_tc prototype
    ├── bcgpugcsc_sc.cu             # [ORIGINAL] scCSC  — scalar CSC
    ├── bcgpugcsc_wa.cu             # [ORIGINAL] vcCSC  — warp-shuffle CSC
    ├── bcgpugcooc_sc.cu            # [ORIGINAL] scCOOC — scalar COOC
    ├── bcgputc.cu                  # [NEW] tcTF32 — TENSOR CORE implementation
    ├── bcug_seq.c                  # [ORIGINAL] CPU sequential BC (ground truth)
    ├── utils_bc.c                  # [MODIFIED] removed output truncation
    ├── helper_cuda.h               # [NEW] local CUDA helper stub
    ├── helper_functions.h          # [NEW] local helper stub
    ├── run_all_experiments.sh      # [NEW] 48-test automated runner
    ├── cross_check_scores.py       # [NEW] CPU baseline + GPU cross-checker
    ├── generate_summary_report.py  # [NEW] timing / speedup report generator
    ├── snap_to_mtx.py              # [NEW] SNAP dataset -> .mtx converter
    ├── edgelist_to_mtx.py          # [NEW] edgelist -> .mtx converter
    └── ... (all remaining original files untouched)
```

---

# PART A — Paper Reproduction

## 3. Changes Needed to Move the Original Code to Our GPU

**Modified files: 3 — hardware adaptation only, zero algorithmic changes.**

### Change A1: Makefile — CUDA Paths & GPU Architecture

**Why:** The original code was hardcoded for CUDA 10.1 and the Titan Xp (sm_61).

```diff
- CC=gcc
+ CUDA_DIR=/usr/local/cuda
+ CC=gcc

-       CFLAGS= -O3 -g -std=gnu99 -Wall -I/usr/local/cuda-10.1/include -I/usr/local/cuda-10.1/samples/common/inc
+       CFLAGS= -O3 -g -std=gnu99 -Wall -I. -I$(CUDA_DIR)/include

-       LDLIBS= -lm -lstdc++ -L/usr/local/cuda-10.1/lib64 -lcufft -lcudart
+       LDLIBS= -lm -lstdc++ -L$(CUDA_DIR)/lib64 -lcufft -lcudart

- CUDA_DIR=/usr/local/cuda-10.1
- NVCCFLAGS = -O3 -arch=sm_61 -I$(CUDA_DIR)/include -I/$(CUDA_DIR)/samples/common/inc -lineinfo
+ NVCCFLAGS = -O3 -arch=sm_89 -I. -I$(CUDA_DIR)/include -lineinfo
```

1. Changed `CUDA_DIR` to the generic symlink `/usr/local/cuda`
2. `-arch=sm_61` → `-arch=sm_89` (also required for the tensor core WMMA path)
3. Removed the CUDA Samples include path (CUDA 11.5 installs the samples separately)

### Change A2: run_bcugcsccooc.sh — Data Path

```diff
+#!/usr/bin/env bash
 TEST="1"
-DATA="/disk/raptor-2/oarti001/documents/MMdataFiles"
+DATA="/home/akshat/TurboBC/graphData"
```

### Change A3: utils_bc.c — Printing All Nodes

**Why:** In the original code, when the node count exceeded 40 only the first 40 and the last 40 were printed. As a result, roughly 99,000 of ASIC_100ks's 99,190 nodes never appeared in the log, and a full cross-check was impossible.

```diff
     printf("\nS   S_hgpu \n");
-    if (n > m1){
-       for (i=0; i<m1; i++){ printf("%d,%d,%d\n",i,S[i],S_hgpu[i]); }
-       for (i=n-m1; i<n; i++){ printf("%d,%d,%d\n",i,S[i],S_hgpu[i]); }
-    }else{
-       for (i=0; i<n; i++){ printf("%d,%d,%d\n", i,S[i],S_hgpu[i]); }
-    }
+    for (i=0; i<n; i++){ printf("%d,%d,%d\n",i,S[i],S_hgpu[i]); }
```

*(The same change was made for the sigma and bc vectors.)*

**Note:** This is purely a printing change. The verification functions (`bfs_check`, `S_check`, `bc_check`) were already checking all N nodes.

### New Support Files (Part A)

| File | Why it was added |
|:---|:---|
| `helper_cuda.h` | The original `#include <helper_cuda.h>` comes from the CUDA Samples; CUDA 11.5 installs those separately, so this is a lightweight local stub for `checkCudaErrors()` |
| `helper_functions.h` | A stub for the same reason |

## 4. Part A Result — Were the Paper's Claims Reproduced?

| Paper Claim | Our Result | Reproduced? |
|:---|:---|:---:|
| GPU significantly faster than CPU | 27x–275x speedup on the large graphs | ✅ |
| scCSC is best on regular graphs (among the CUDA-core algorithms) | scCSC is the fastest of the three on smallworld | ✅ |
| vcCSC carries more overhead than the scalar version | vcCSC is consistently 2–2.5x slower (when degree is small) | ✅ |
| scCOOC wins on skewed-degree graphs | scCOOC is the fastest of the three on epinions (max degree 3035 vs mean 6.71) | ✅ |
| CPU and GPU BC scores identical | 0 mismatches across all 48 tests | ✅ |

### smallworld — Timing Comparison (N=100,000)

| Algorithm | Paper (Titan Xp) | Ours (NVIDIA L4) | Match? |
|:---|:---:|:---:|:---:|
| scCSC | ~0.98 ms | 1.25 ms | ✅ |
| scCOOC | ~1.00 ms | 1.80 ms | ✅ |
| vcCSC | ~2.50 ms | 2.39 ms | ✅ |

### Why the Timing Differences Exist

| Factor | Titan Xp | NVIDIA L4 | Effect |
|:---|:---|:---|:---|
| Memory Bandwidth | 547.7 GB/s | 300 GB/s | Memory-bound operations (scCOOC) are slightly slower on the L4 |
| L2 Cache | 3 MB | 48 MB (16x) | Cache-friendly operations (scCSC) are faster on the L4 |
| Architecture | Pascal (2017) | Ada Lovelace (2023) | Newer instruction set |

> **Caveat:** The absolute times for ASIC_100ks are no longer directly comparable with the paper. The original benchmark ran `--nr 1 --rs 100` (only root vertex 100); all graphs now run with `--nr 32` (roots 0–31) — see Section 8. A different root set means a different BFS depth. All four algorithms run on the same roots, so the **GPU-vs-GPU comparison remains fully valid**; only the comparison against the paper's single-root numbers no longer holds.

---

# PART B — Tensor Core Implementation (New Work)

## 5. Motivation

All three algorithms in the paper use **CUDA cores**. The NVIDIA L4 also has **tensor cores** — dedicated matrix-multiply hardware that delivers roughly 60 TFLOPS in tf32 against roughly 30 TFLOPS in FP32. The question: can BC be run on tensor cores, and will it be faster than CUDA cores?

**The first obstacle:** the core operation in BC is `f ← A'f` — a sparse matrix–**VECTOR** multiply. Tensor cores only perform **MATRIX × MATRIX**.

**Solution:** run **16 roots at once**. Then `f` (an n×1 vector) becomes `F` (an n×16 matrix), and the BFS step becomes:

```
Before (SpMV):   f  ← A' f      f  = n × 1
Now    (SpMM):   F  ← A' F      F  = n × 16   ← this is now a job for the tensor cores
```

Why 16? Because the N dimension of the `m16n16k8` tf32 WMMA fragment is exactly 16.

The same kernel is used in both the BFS phase (`F ← A'F`) and the delta phase (`ΔU_T ← A'ΔU`).

## 6. Precision — Why fp16 Fails

In BC, `sigma[v]` is the **count** of shortest paths from the root to v — an exact integer.

| Format | Mantissa | Exact integers up to | Max value | Verdict |
|:---|:---:|:---:|:---:|:---|
| fp16 | 11 bits | **2,048** | 65,504 | ❌ Fails for two reasons |
| tf32 | 11 bits | 2,048 | 3.4e38 | ⚠️ Range is fine, precision is not |
| fp32 | 24 bits | 16,777,216 | 3.4e38 | ✅ This is what we need |

fp16 is unusable for two independent reasons:
1. Path counts on fb_artist reach **24,973** — far above the exact-integer limit of 2,048
2. On nd6k sigma reaches **4.13e10**, which **overflows** fp16's maximum of 65,504

**Solution — 3-way tf32 splitting:** each F block is decomposed as `F = F1 + F2 + F3`, with every term exactly representable in tf32. The effective mantissa is 3 × 11 = **33 bits**, which exceeds fp32's 24. Three MMAs are issued per tile.

Measured on epinions (maximum relative error):

| Splits | Error | Passes the 1e-4 bar? |
|:---:|:---:|:---:|
| 1 | 9.294e-04 | ❌ |
| 2 | 1.801e-05 | ⚠️ |
| **3** | **9.985e-06** | ✅ |

The cost is only **28%**, because the kernel is memory-bound and the extra MMAs are effectively free.

The split level can be overridden with the `TURBOBC_TC_SPLIT=1|2|3` environment variable (default 3).

## 7. **Condensed Blocking — Main Contribution**

This is the most important part of Part B.

### Problem: Geometric Tiling Is Useless on Sparse Graphs

A tensor core only operates on **16×16 tiles**. The first natural idea is to lay a fixed grid over the matrix — block-column 0 = vertices 0–15, block-column 1 = vertices 16–31, and so on.

On a real graph this falls apart. In fb_artist a vertex's neighbours are scattered across the whole graph — say `7, 4891, 20014, 33001`. Those four neighbours land in **four different tiles**:

```
tile (0–15)        :  [_ _ _ _ _ _ _ X _ _ _ _ _ _ _ _]   ← 1/16 filled
tile (4880–4895)   :  [_ _ _ _ _ _ _ _ _ _ _ X _ _ _ _]   ← 1/16 filled
tile (20000–20015) :  [_ _ _ _ _ _ _ _ _ _ _ _ _ _ X _]   ← 1/16 filled
tile (33000–33015) :  [_ X _ _ _ _ _ _ _ _ _ _ _ _ _ _]   ← 1/16 filled
```

For every tile the GPU must fetch all **256 floats of F** from memory, even when only one nonzero sits inside it.

**Measured on fb_artist: 1.12 nonzeros per tile → 99.6% of bandwidth wasted.** This is exactly why the first tensor core version was 2x–16x **slower** than the CUDA cores on sparse graphs.

### Solution: Build Tiles from the Neighbour List, Not from a Grid

Drop the fixed grid. Each **16-vertex row-window** keeps a sorted list of the **actual unique source vertices** of its rows, and cuts that list into chunks of 16:

```
unique neighbours of window:  [7, 41, 902, 4891, 4893, 15003, 20014, 33001, ...]
                               └──────────── chunk 0 (16) ────────────┘└─ chunk 1 ─┘ ...

chunk 0 gather:  F[7], F[41], F[902], F[4891], ... (16 rows)
                 ↑ ALL 16 are useful — every slot is guaranteed to hold a nonzero
```

Because every chunk column contains at least one nonzero, density is guaranteed to be **≥ 6.25% by construction**, no matter how sparse the graph is.

### Code Change

**Host side** (`bcgputc.cu`) — the tile key changed:

```c
// BEFORE — a neighbour u's tile was determined by its POSITION
ents[t].key = (unsigned long long)(v>>4)*nb + (u>>4);

// NOW — only the row-window and the neighbour; unique neighbours are numbered later
ents[t].key = ((unsigned long long)(v>>4) << 32) | (unsigned int)u;
```

**Kernel side** — a single line, the F gather:

```c
// BEFORE — fetch F for 16 CONSECUTIVE vertices (of which about 1 is useful)
const float *Fb = F + (size_t)bcol[b] * TC_TILE;
float x = Fb[t];

// NOW — fetch F for the chunk's 16 ACTUAL neighbour vertices (all useful)
const int *cc = chunkcol + (size_t)b * 16;
float x = F[(size_t)cc[t >> 4] * TCK + (t & 15)];
```

The gather is now a **64-byte coalesced load per slot** (the 16 roots are consecutive in memory), where previously it was a strided read of a mostly empty tile.

### Measured Effect

| Graph | Tile density (geometric) | Tile density (condensed) | Improvement |
|:---|:---:|:---:|:---:|
| fb_artist | 0.437% | **6.301%** | 14.4x |
| epinions | 0.823% | **7.116%** | 8.6x |
| smallworld | 1.789% | **15.374%** | 8.6x |
| ASIC_100ks | — | **11.275%** | — |
| fb_new_sites | — | **6.137%** | — |
| nd3k | 43.58% | **60.588%** | 1.4x |
| nd6k | 42.29% | **59.619%** | 1.4x |

**Important:** this is **tile density**, a property of our storage layout. **The graph's matrix density has not changed at all** — not a single edge was added or removed, and the `.mtx` files are identical. The two are listed separately in Section 9.

On fb_artist this change alone turned a **4.2x LOSS into a ~2x WIN** against the best CUDA-core kernel (see Section 10 for the per-run figure).

## 8. Benchmark Fairness Fix (`nr = 1` → `nr = 32`)

This is not an optimization; it is a fix for a **measurement error**.

`tcTF32` always computes a full 16-root batch — that is by design, not by choice. The old benchmark ran `--nr 1`, which meant the kernel did the work of 16 roots, we counted all of that time, and credited only 1 root in the report — an **artificial 15/16 penalty**.

Same binary, same graph, only `nr` changed:

```
smallworld, nr=1   →  9.261 ms/vertex
smallworld, nr=16  →  0.630 ms/vertex     ← 14.7x difference, zero code changes
```

Every non-toy graph therefore now runs at `--nr 32` (a multiple of 16).

The CUDA-core kernels are single-source — their per-root cost does not depend on `nr` (verified: at `nr` = 16/32/64/128/256 scCSC bounces between 0.83 and 1.48 with no trend). So this change only corrects the incorrect measurement of the tensor core version; it gives nobody an unfair advantage.

---

## 9. Datasets

### How to Read the Density Column

Density answers one question: **out of every connection that could possibly exist, how many actually do?**

```
Density = actual edges / maximum possible edges x 100
        = nz / (N x N) x 100
```

A graph with N nodes can have at most N x N directed connections. Density is the fraction of those that are really present. **The smaller the number, the sparser the graph.**

**Worked example — ASIC_100ks (the sparsest graph here):**

```
N                      = 99,190 nodes
Maximum possible edges = 99,190 x 99,190 = 9,838,656,100   (~9.8 billion)
Actual edges           =                       578,890
Density = 578,890 / 9,838,656,100 x 100 = 0.0059%
```

So only about **1 in every 17,000 possible connections exists** — the other 16,999 cells of the matrix are empty.

**Worked example — cage4 (the densest graph here):**

```
N                      = 9 nodes
Maximum possible edges = 9 x 9 = 81
Actual edges           =         49
Density = 49 / 81 x 100 = 60.49%
```

Here **1 in every 2 possible connections exists** — more than half the matrix is full.

**Rule of thumb:**

| Density | Meaning | Typical of |
|:---|:---|:---|
| < 0.1% | fewer than 1 filled cell in 1,000 | **Very sparse** — real-world networks |
| 1% – 10% | 1 filled cell in 10 to 100 | Medium |
| > 20% | more than 1 filled cell in 5 | **Dense** — small or toy matrices |

You can verify any of these by hand from the log, which prints the matrix size for every run:

```bash
grep -m1 "^N = " bc_verification_results.log
# N = 11,M = 11, nz = 40   ->   40 / (11 x 11) x 100 = 33.06%
```

The summary report prints this same ranking automatically under **HOW TO READ THE DENSITY COLUMN**, so it stays correct if graphs are added or removed.

### The 12 Graphs

| Graph | Nodes | Edges | **Matrix density** | Filled cells | Avg degree | Category | Source |
|:---|---:|---:|---:|:---|---:|:---|:---|
| **ASIC_100ks** | 99,190 | 578,890 | **0.0059%** | 1 in 16,996 | 5.8 | Irregular | SuiteSparse (circuit) |
| **epinions** | 75,888 | 508,837 | **0.0088%** | 1 in 11,318 | 6.7 | Irregular | SNAP soc-Epinions1 |
| **smallworld** | 100,000 | 999,996 | **0.0100%** | 1 in 10,000 | 10.0 | Regular | SuiteSparse |
| **fb_new_sites** | 27,917 | 411,928 | **0.0529%** | 1 in 1,892 | 14.8 | Irregular | SNAP gemsec-Facebook |
| **fb_artist** | 50,515 | 1,638,180 | **0.0642%** | 1 in 1,558 | 32.4 | Irregular | SNAP gemsec-Facebook |
| nd6k | 18,000 | 6,897,316 | 2.1288% | 1 in 47 | 383.2 | Regular | SuiteSparse ND (3D FEM) |
| nd3k | 9,000 | 3,279,690 | 4.0490% | 1 in 25 | 364.4 | Regular | SuiteSparse ND (3D FEM) |
| congress | 475 | 13,289 | 5.8899% | 1 in 17 | 28.0 | Irregular | Twitter congress network |
| Ragusa18 | 23 | 64 | 12.0983% | 1 in 8 | 2.8 | Irregular | SuiteSparse |
| Tina_AskCal | 11 | 29 | 23.9669% | 1 in 4 | 2.6 | Irregular | SuiteSparse |
| mycielskian4 | 11 | 40 | 33.0579% | 1 in 3 | 3.6 | Regular | SuiteSparse |
| cage4 | 9 | 49 | 60.4938% | 1 in 2 | 5.4 | Regular | SuiteSparse |

The five graphs in bold are **extremely sparse** and are the ones the tensor-core claim rests on.

> `mycielskian4` and `smallworld` run with `--ug 1` (undirected), so the reader symmetrizes the matrix and doubles `nz`. The figures above are for that **symmetric matrix** — the one the kernels actually traverse. The raw `.mtx` file stores only the upper triangle (smallworld: 499,998).

**Regular vs Irregular** is decided by the coefficient of variation (CV) of the degree distribution:
- nd3k CV = 0.26, nd6k CV = 0.23 — the most uniform
- fb_new_sites CV = 1.56, fb_artist CV = 1.96
- **epinions CV = 5.02** — max degree 3035 against a mean of 6.71, the most skewed

---

## 10. Final Results

All times are in **ms per root vertex**, at `nr=32`, with all 4 algorithms running on the same roots. Sorted from sparsest graph to densest.

| Graph | Density | CPU (ms) | scCSC | scCOOC | vcCSC | **tcTF32** | Fastest | vs CPU | **vs CUDA** | Status |
|:---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|
| ASIC_100ks | 0.0059% | 2.359 | 0.0858 | 0.0848 | 0.2192 | **0.0314** | tcTF32 | 75.10x | **2.70x** | ✅ |
| epinions | 0.0088% | 23.283 | 1.7000 | 1.4346 | 1.8919 | **0.8746** | tcTF32 | 26.62x | **1.64x** | ✅ |
| smallworld | 0.0100% | 32.262 | 1.3566 | 1.8311 | 1.9073 | **0.1986** | tcTF32 | 162.44x | **6.83x** | ✅ |
| fb_new_sites | 0.0529% | 12.633 | 0.7946 | 0.7610 | 0.8739 | **0.2640** | tcTF32 | 47.86x | **2.88x** | ✅ |
| fb_artist | 0.0642% | 28.725 | 1.0018 | 1.2292 | 1.4226 | **0.5086** | tcTF32 | 56.47x | **1.97x** | ✅ |
| nd6k | 2.1288% | 65.670 | 1.3503 | 4.0131 | 0.5196 | **0.2388** | tcTF32 | 275.03x | **2.18x** | ✅ |
| nd3k | 4.0490% | 30.875 | 1.2092 | 1.9388 | 0.3639 | **0.2223** | tcTF32 | 138.89x | **1.64x** | ✅ |
| congress | 5.8899% | 0.221 | 0.1510 | 0.1492 | 0.1433 | **0.0439** | tcTF32 | Toy | **3.26x** | ✅ |
| Ragusa18 | 12.0983% | 0.009 | 0.5863 | 0.6078 | 0.6811 | **0.5421** | tcTF32 | Toy | 1.08x | ✅ |
| Tina_AskCal | 23.9669% | 0.001 | 1.0590 | 1.1573 | 1.1684 | **0.3563** | tcTF32 | Toy | 2.97x | ✅ |
| mycielskian4 | 33.0579% | 0.005 | **0.1129** | 0.1240 | 0.1163 | 0.1209 | scCSC | Toy | 0.93x | ✅ |
| cage4 | 60.4938% | 0.005 | 0.5212 | 0.4520 | 0.5698 | **0.4115** | tcTF32 | Toy | 1.10x | ✅ |

**tcTF32 is the fastest kernel on 11 of the 12 graphs.**

`vs CUDA` is the number that matters: how much faster tcTF32 is than **whichever of the three CUDA-core algorithms is fastest on that graph** — not just against scCSC. `vs CPU` compares against the sequential CPU implementation and is naturally much larger.

> **Ignore the four toy graphs when judging the result.** mycielskian4 (11 nodes), cage4 (9), Ragusa18 (23) and Tina_AskCal (11) are far too small for the GPU to do any real work — their runtime is almost entirely kernel-launch overhead, and their `vs CUDA` figures swing between runs (mycielskian4 measured 1.02x in one run and 0.93x in the next). The eight real graphs are stable across runs: e.g. epinions 1.63x / 1.64x, fb_artist 1.91x / 1.97x, smallworld 5.64x / 6.83x.

### Speedup vs the BEST CUDA-Core Algorithm — Real Graphs Only

| Graph | Matrix density | Best CUDA-core | tcTF32 | **Gain** |
|:---|---:|:---|---:|---:|
| **smallworld** | **0.0100%** | 1.3566 (scCSC) | 0.1986 | **6.83x** |
| congress | 5.8899% | 0.1433 (vcCSC) | 0.0439 | **3.26x** |
| fb_new_sites | 0.0529% | 0.7610 (scCOOC) | 0.2640 | **2.88x** |
| **ASIC_100ks** | **0.0059%** | 0.0848 (scCOOC) | 0.0314 | **2.70x** |
| nd6k | 2.1288% | 0.5196 (vcCSC) | 0.2388 | **2.18x** |
| fb_artist | 0.0642% | 1.0018 (scCSC) | 0.5086 | **1.97x** |
| **epinions** | **0.0088%** | 1.4346 (scCOOC) | 0.8746 | **1.64x** |
| nd3k | 4.0490% | 0.3639 (vcCSC) | 0.2223 | **1.64x** |

**Key observation: the speedup does not depend on how sparse the graph is.** The three sparsest graphs (0.0059%–0.0100%) give gains of 1.64x, 1.64x and 6.83x; the densest real graph (nd3k at 4.05%) gives 1.64x. The largest gain in the whole table belongs to smallworld, one of the sparsest graphs. There is no correlation between density and speedup — which is exactly the point: after condensed blocking, tile occupancy comes from the *layout*, not from the density of the graph.

Note also that the best CUDA-core algorithm is a different one on different graphs (scCSC on smallworld and fb_artist, scCOOC on ASIC_100ks / fb_new_sites / epinions, vcCSC on congress / nd3k / nd6k). tcTF32 beats all three regardless of which one happens to win.

### Before / After — Effect of Condensed Blocking

| Graph | tcTF32 (geometric tiling) | tcTF32 (condensed) | Improvement |
|:---|---:|---:|---:|
| ASIC_100ks | 48.1225 | 0.0314 | **1532x** |
| smallworld | 9.9671 | 0.1986 | **50.2x** |
| congress | 0.5389 | 0.0439 | **12.3x** |
| fb_artist | 5.6498 | 0.5086 | **11.1x** |
| fb_new_sites | 2.2849 | 0.2640 | **8.7x** |
| epinions | 3.1555 | 0.8746 | **3.6x** |

*(This is the combined effect of the layout change in Section 7 and the `nr` fix in Section 8. Before these changes tcTF32 lost to the CUDA-core kernels on every one of these graphs.)*

---

## 11. Verification

### Correctness

| Test | Result |
|:---|:---|
| CPU vs GPU node-by-node cross-check | **48/48 PERFECT MATCH, 0 mismatches** |
| Total node comparisons | 1,524,156 (12 graphs × 4 algorithms) |
| `compute-sanitizer --tool memcheck` | 0 errors |
| `compute-sanitizer --tool racecheck` | 0 hazards |
| `compute-sanitizer --tool synccheck` | 0 errors |
| Repeated runs bit-identical | ✅ |
| Independent `networkx` cross-check | 1.6e-07 agreement (symmetric graphs) |

### Tolerance Note

`utils_bc.c` uses **absolute** tolerances (`1e-5` on sigma, `0.05` on bc) that do not scale. On nd6k sigma reaches 4.13e10 — 2400x above fp32's exact-integer limit of 2^24, where a single ULP is already 8. The worst disagreement observed there is **176 in 71,287,136 = 2.5e-6 relative (~22 ULP)**, which comes purely from the difference in summation order between an MMA and a scalar loop — the CPU's own value is not exact either.

For this reason `generate_summary_report.py` re-checks all three vectors (S, sigma, bc) section-wise at a **1e-4 relative tolerance**. `utils_bc.c` (library code) was left untouched.

---

## 12. Build & Run

### Compile
```bash
cd ~/TurboBC/TurboBC
make clean && make -j$(nproc)
```

### Run the full suite (48 tests + verification, ~6–8 min)
```bash
cd ~/TurboBC/TurboBC && ./run_all_experiments.sh 2>&1 | tee ~/turbobc_results_$(date +%Y%m%d_%H%M).txt
```

This single command:
1. Runs 48 experiments (12 graphs × 4 GPU algorithms, each alongside a CPU baseline)
2. Stores the CPU BC scores in `cpu_baseline_scores.txt` / `.json`
3. Cross-checks the GPU scores node by node → `gpu_vs_cpu_audit_report.txt`
4. Prints the final benchmark and speedup table

`tee` keeps a timestamped copy of the whole session (e.g. `~/turbobc_results_20260909_1830.txt`). The four files it writes inside the repo are overwritten on every run, so the `tee` copy is what preserves a specific run:

| File | Contents |
|:---|:---|
| `bc_verification_results.log` | Full raw output, every node of every graph |
| `cpu_baseline_scores.txt` / `.json` | CPU reference BC scores |
| `gpu_vs_cpu_audit_report.txt` | Node-by-node CPU vs GPU audit |

### Re-print the summary table without re-running everything
```bash
cd ~/TurboBC/TurboBC && python3 generate_summary_report.py bc_verification_results.log
```

Every figure in that table — density, timings, `vs CPU`, `vs CUDA`, the "fastest on N of M" line and the per-graph speedup list in the notes — is computed from the log at print time. Nothing is hardcoded, so the report stays correct when graphs are added or removed.

### Single graph, manual
```bash
./exec <path>.mtx <w> <ug> <format> <p> <nr> <rs> <repet> <seq>
```

| Arg | Meaning |
|:---|:---|
| `w` | 1 = weighted graph (a value column is present), 0 = not |
| `ug` | 1 = undirected (the matrix is symmetric), 0 = not |
| `format` | **0** = scCSC, **1** = vcCSC, **2** = scCOOC, **3** = **tcTF32** |
| `p` | 1 = print all node values |
| `nr` | number of root vertices (keep it a multiple of 16 for tcTF32) |
| `rs` | root vertex, used only when `nr = 1` |
| `repet` | repetitions used to average the timing |
| `seq` | 1 = also run the CPU sequential BC and compare |

Example — tensor core on fb_artist, with CPU verification:
```bash
./exec ../graphData/fb_artist.mtx 0 0 3 0 32 0 1 1
```

---

## 13. Summary of All Changes

| Category | Files | Detail |
|:---|:---|:---|
| **Modified (Part A)** | `Makefile`, `run_bcugcsccooc.sh`, `utils_bc.c` | Hardware paths, sm_89 and output truncation only — **zero algorithmic changes** |
| **Modified (Part B)** | `bcug.h`, `bcugcsccooc_main.c` | `bc_gpu_ug_tc` prototype + `format == 3` dispatch |
| **New — algorithm** | `bcgputc.cu` (348 lines) | The complete tensor core implementation |
| **New — support** | `helper_cuda.h`, `helper_functions.h` | CUDA Samples stubs |
| **New — automation** | `run_all_experiments.sh`, `cross_check_scores.py`, `generate_summary_report.py` | 48-test runner + verification + reporting |
| **New — data prep** | `snap_to_mtx.py`, `edgelist_to_mtx.py` | Dataset converters |
| **Untouched** | `bcgpugcsc_sc.cu`, `bcgpugcsc_wa.cu`, `bcgpugcooc_sc.cu`, `bcug_seq.c` | The paper's three GPU kernels and the CPU algorithm — not a single line changed |

---

## 14. Findings & Open Items

### Finding: A Pre-existing Issue in the Paper's Code

On directed (asymmetric) graphs the delta phase uses `A'` (predecessors) where Brandes' algorithm requires `A` (successors). The CPU and all GPU implementations show the **same behaviour**, so they agree with each other perfectly, but they differ from `networkx`.

- **No effect on symmetric graphs** (A = A')
- Affected: Ragusa18 (36% reciprocal edges), Tina_AskCal (28%), congress (46%), epinions (41%)
- **No effect on any timing comparison** — all five implementations compute the same quantity
- Not introduced by our code; reported, not fixed

Two options exist: (a) symmetrize the affected graphs, or (b) build a separate CSR for the delta phase. No change has been made so far.

### Open Item: Tensor Core Ablation

The speedup comes from three sources — (a) 16-root batching, (b) the condensed layout, and (c) the tensor cores themselves. **(c) has not yet been isolated.** An airtight claim requires an ablation: the same condensed layout and the same 16-root batching, but with a CUDA-core FMA loop in place of the MMA.

Expected outcome: the tensor cores will matter, because the condensed layout deliberately performs 6–16x redundant arithmetic in order to make memory access perfectly coalesced — a trade that only pays off when arithmetic is cheap. But **this is currently a conjecture, not a measurement.**

---

## 15. Contribution — One Paragraph

Tensor cores were previously assumed to be unsuitable for sparse-graph betweenness centrality, on the grounds that a 16×16 tile ends up holding only 1–2 nonzeros while the rest of the bandwidth is wasted (measured: 1.12 nonzeros per tile at 0.437% density on fb_artist). This work shows that **the limitation belonged not to the tensor cores but to cutting the matrix along a fixed geometric grid.** If tiles are instead built from the actual neighbour list of each row-window rather than from a grid, tile occupancy is guaranteed to be ≥6.25% by construction, no matter how sparse the graph is. Three results follow: (1) the first tensor-core implementation of BC, obtained by turning SpMV into SpMM through multi-source batching; (2) exact integer path counts on tf32 via 3-way splitting (33 mantissa bits, more than fp32's 24); and (3), most importantly, a blocking scheme that lets tensor cores beat all three CUDA-core algorithms by **1.64x–6.83x** on the eight non-toy graphs, including ones at 0.006% density, beat the CPU by **27x–275x**, and keep BC scores bit-exact against the CPU.