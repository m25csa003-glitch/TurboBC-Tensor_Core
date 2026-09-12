# TurboBC — Tensor Core Extension

A **tensor-core implementation of Betweenness Centrality** (`tcTF32`), built on top of the TurboBC algorithms of Artiles & Saeed (ICPP Workshops '21).

This repository contains two things:

* **Part A — a reproduction** of the original paper's three CUDA-core algorithms on modern hardware (NVIDIA L4, Ada Lovelace, sm_89), with all of its scientific claims verified
* **Part B — a new fourth algorithm** that runs the BC computation on the GPU's **tensor cores** instead of its CUDA cores, and beats all three original algorithms on every real graph tested — including graphs at 0.006% density

---

## The Headline Result

Tensor cores were generally assumed to be a poor fit for sparse-graph betweenness centrality: a 16×16 tile of a real adjacency matrix holds only 1–2 non-zeros, so most of the memory bandwidth is wasted. On `fb_artist` we measured **1.12 non-zeros per tile — a tile occupancy of 0.437%** — and the first version of our tensor-core kernel duly lost to the CUDA-core kernels on every sparse graph.

**That limitation turned out to belong to the *tiling*, not to the tensor cores.**

Cutting the matrix along a fixed geometric grid is what leaves the tiles empty, because a vertex's neighbours are scattered across the whole graph. If tiles are instead built from **each row-window's actual neighbour list**, every column of a tile is guaranteed to carry a non-zero, and occupancy is bounded below by 6.25% by construction — no matter how sparse the graph is.

| Graph | Geometric tiling | Condensed blocking |
|:---|---:|---:|
| fb_artist | 0.437% | **6.301%** |
| epinions | 0.823% | **7.116%** |
| smallworld | 1.789% | **15.374%** |

With that change, `tcTF32` becomes the fastest of the four kernels on 11 of 12 test graphs.

---

## Results

Times are **ms per root vertex**, at `nr = 32`, all four algorithms on identical roots. Sorted sparsest graph first.

| Graph | Density | Nodes | CPU | scCSC | scCOOC | vcCSC | **tcTF32** | **vs best CUDA core** |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| ASIC_100ks | 0.0059% | 99,190 | 2.359 | 0.0858 | 0.0848 | 0.2192 | **0.0314** | **2.70x** |
| epinions | 0.0088% | 75,888 | 23.283 | 1.7000 | 1.4346 | 1.8919 | **0.8746** | **1.64x** |
| smallworld | 0.0100% | 100,000 | 32.262 | 1.3566 | 1.8311 | 1.9073 | **0.1986** | **6.83x** |
| fb_new_sites | 0.0529% | 27,917 | 12.633 | 0.7946 | 0.7610 | 0.8739 | **0.2640** | **2.88x** |
| fb_artist | 0.0642% | 50,515 | 28.725 | 1.0018 | 1.2292 | 1.4226 | **0.5086** | **1.97x** |
| nd6k | 2.1288% | 18,000 | 65.670 | 1.3503 | 4.0131 | 0.5196 | **0.2388** | **2.18x** |
| nd3k | 4.0490% | 9,000 | 30.875 | 1.2092 | 1.9388 | 0.3639 | **0.2223** | **1.64x** |
| congress | 5.8899% | 475 | 0.221 | 0.1510 | 0.1492 | 0.1433 | **0.0439** | **3.26x** |

**1.64x – 6.83x faster than the best CUDA-core kernel** on every graph, and **26x – 275x faster than the sequential CPU implementation**.

Note that the *best* CUDA-core kernel differs by graph — `scCSC` on smallworld and fb_artist, `scCOOC` on ASIC_100ks / fb_new_sites / epinions, `vcCSC` on congress / nd3k / nd6k. `tcTF32` beats all three regardless of which one happens to win.

**The speedup does not correlate with density.** The largest gain in the table (6.83x) belongs to `smallworld`, one of the sparsest graphs; the densest real graph gives 1.64x. After condensed blocking, tile occupancy is a property of the layout, not of the graph.

*(Four toy graphs in the suite — 9 to 23 nodes — are omitted above. They are too small for the GPU to do meaningful work and their timings are dominated by kernel-launch overhead.)*

### Correctness

Every BC score is checked node by node against the sequential CPU implementation:

```
48 / 48 tests                    ✅ PERFECT MATCH
1,524,156 node comparisons       0 mismatches
compute-sanitizer memcheck       0 errors
compute-sanitizer racecheck      0 hazards
compute-sanitizer synccheck      0 errors
independent networkx cross-check 1.6e-07 agreement
```

---

## The Four Algorithms

Selected with the `--format` flag. All four compute the same quantity; they differ in how the dominant operation is mapped onto the hardware.

| `--format` | Name | Parallelisation | Best when |
|:---:|:---|:---|:---|
| `0` | `scCSC` | one **thread per vertex** | degrees are roughly uniform |
| `1` | `vcCSC` | one **warp (32 threads) per vertex** | a few vertices have very high degree |
| `2` | `scCOOC` | one **thread per edge** | the degree distribution is heavily skewed |
| `3` | **`tcTF32`** | **tensor cores, 16 roots at a time** | **every graph tested** |

Formats 0–2 are the original paper's kernels, unmodified. Format 3 is new.

### How the tensor-core version works, in brief

A tensor core can only multiply **matrices**, never vectors. The BC inner loop is an SpMV:

```
f  ←  A' f                 f is n × 1   →   a tensor core cannot execute this
```

So `tcTF32` processes **16 root vertices simultaneously**, which turns the frontier into a matrix and the operation into an SpMM:

```
F  ←  A' F                 F is n × 16  →   exactly the m16n16k8 tf32 WMMA shape
```

16 is not arbitrary — it is the N dimension of the fragment, so no output lane is wasted.

Two problems had to be solved:

1. **Exactness.** Shortest-path counts are integers reaching 4.13 × 10¹⁰. fp16 fails twice over (exact integers stop at 2,048; `sigma` would overflow its 65,504 maximum). TF32 has fp32's range but only an 11-bit mantissa, so each value is split three ways — 33 mantissa bits, more than fp32's own 24 — at a measured cost of about 20%.
2. **Sparsity.** Solved by the condensed blocking described above.

**The design never stores a dense matrix.** The graph is held in a sparse condensed format — O(nz), 2.3–8.2 MB for these graphs — and a 16×16 dense window is assembled in *shared memory* for each chunk, multiplied, and discarded immediately, around 3.25 million times per run. A dense `n × n` matrix would be 37 GB for `smallworld` and would not even fit on the GPU.

📖 **Full walkthrough:** [`TurboBC/TENSOR_CORE_DESIGN.md`](TurboBC/TENSOR_CORE_DESIGN.md) — the three stages and their lifetimes, exact memory accounting, scaling limits, precision analysis, and commands to reproduce every figure.

📖 **Reproduction study:** [`TurboBC/REPRODUCTION_REPORT.md`](TurboBC/REPRODUCTION_REPORT.md) — what changed to port the original code to sm_89, verification of the paper's claims, and the full dataset description.

---

## Prerequisites

Tested on:

* CUDA 11.5, NVIDIA L4 (Ada Lovelace, sm_89, 23 GB, 58 SMs)
* gcc 11.4.0
* Ubuntu 22.04 (kernel 5.15)
* Python 3.10 (for the verification and reporting scripts)

Tensor cores require **sm_70 or newer**; the TF32 path used here requires **sm_80 or newer**. Formats 0–2 run on any CUDA-capable GPU.

The original paper's environment was CUDA 10.1 on an NVIDIA Titan Xp (sm_61) — which has no tensor cores, so Part B was not possible on that hardware.

---

## Install

```bash
git clone https://github.com/m25csa003-glitch/TurboBC-Tensor_Core.git
cd TurboBC-Tensor_Core/TurboBC
make clean && make -j$(nproc)
```

If your CUDA lives somewhere other than `/usr/local/cuda`, or your GPU is not sm_89, edit `CUDA_DIR` and `-arch=` at the top of the `Makefile`.

---

## Run

### Full benchmark suite — 48 tests with verification (~7 min)

```bash
cd TurboBC
./run_all_experiments.sh 2>&1 | tee ~/turbobc_results_$(date +%Y%m%d_%H%M).txt
```

This one command runs all 12 graphs × 4 GPU algorithms, each alongside a CPU baseline, cross-checks every BC score node by node, and prints the benchmark table. Every figure in that table is computed from the log at print time — nothing is hardcoded.

### Re-print the summary without re-running

```bash
python3 generate_summary_report.py bc_verification_results.log
```

### A single graph

```bash
./exec <path>.mtx <w> <ug> <format> <p> <nr> <rs> <repet> <seq>
```

| Argument | Meaning |
|:---|:---|
| `w` | 1 if the `.mtx` has a value column, 0 otherwise |
| `ug` | 1 if the graph is undirected (matrix symmetric), 0 otherwise |
| `format` | **0** = scCSC, **1** = vcCSC, **2** = scCOOC, **3** = **tcTF32** |
| `p` | 1 to print every node's values |
| `nr` | number of root vertices — **keep this a multiple of 16 for `tcTF32`** |
| `rs` | root vertex, used only when `nr = 1` |
| `repet` | repetitions, for averaging the timing |
| `seq` | 1 to also run the sequential CPU BC and compare |

Example — tensor cores on `fb_artist`, with CPU verification:

```bash
./exec ../graphData/fb_artist.mtx 0 0 3 0 32 0 1 1
```

> ⚠️ `tcTF32` always computes a full 16-root batch, because that is its design rather than a choice. Timing it at `nr = 1` charges it for 16 roots and credits it with one — an artificial 15/16 penalty. On `smallworld` the same binary reports 9.26 ms/vertex at `nr = 1` and 0.63 ms/vertex at `nr = 16`. Always use a multiple of 16.

Optional environment variable:

```bash
TURBOBC_TC_SPLIT=2   # tf32 split level: 1, 2 or 3 (default 3)
```

Three splits give 33 mantissa bits, which is a *proof* of exactness. Two give 22 bits — empirically sufficient on every graph here, and 1.03x–1.14x faster, but without a guarantee. See Section 13 of the design document.

---

## Datasets

12 graphs in `graphData/`, spanning four orders of magnitude of density:

| Graph | Nodes | Edges | Density | Source |
|:---|---:|---:|---:|:---|
| ASIC_100ks | 99,190 | 578,890 | 0.0059% | SuiteSparse (circuit) |
| epinions | 75,888 | 508,837 | 0.0088% | SNAP `soc-Epinions1` |
| smallworld | 100,000 | 999,996 | 0.0100% | SuiteSparse |
| fb_new_sites | 27,917 | 411,928 | 0.0529% | SNAP `gemsec-Facebook` |
| fb_artist | 50,515 | 1,638,180 | 0.0642% | SNAP `gemsec-Facebook` |
| nd6k | 18,000 | 6,897,316 | 2.1288% | SuiteSparse ND (3D FEM) |
| nd3k | 9,000 | 3,279,690 | 4.0490% | SuiteSparse ND (3D FEM) |
| congress | 475 | 13,289 | 5.8899% | Twitter congress network |
| Ragusa18, Tina_AskCal, mycielskian4, cage4 | 9–23 | 20–64 | 12–60% | SuiteSparse (toy) |

**Density** is `nz / N² × 100` — what fraction of all possible connections actually exists. Lower means sparser: ASIC_100ks has about **1 filled cell in every 17,000**.

Two converters are included for adding more graphs:

```bash
# SNAP datasets (.txt, .txt.gz or .csv); --undirected emits both (u,v) and (v,u)
python3 snap_to_mtx.py  input.txt.gz  output.mtx  [--undirected]

# plain edge lists; --unweighted drops the value column
python3 edgelist_to_mtx.py  input.edgelist  output.mtx  [--unweighted] [--symmetrize]
```

Both emit 1-based, column-grouped Matrix Market files, which is what the reader in `bcugcsccooc_main.c` expects. Graphs converted with `--undirected` already contain both directions, so run them with `--ug 0`.

More matrices can be taken from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu) or [SNAP](https://snap.stanford.edu/data/).

---

## Repository Layout

```
graphData/                       12 graphs in Matrix Market format
TurboBC/
├── bcgputc.cu                   ★ NEW — the tensor core implementation
├── bcgpugcsc_sc.cu              original — scCSC
├── bcgpugcsc_wa.cu              original — vcCSC
├── bcgpugcooc_sc.cu             original — scCOOC
├── bcug_seq.c                   original — sequential CPU BC (ground truth)
├── bcugcsccooc_main.c           entry point
├── TENSOR_CORE_DESIGN.md        ★ full design walkthrough
├── REPRODUCTION_REPORT.md       ★ reproduction study + results
├── run_all_experiments.sh       ★ 48-test runner
├── cross_check_scores.py        ★ node-by-node CPU vs GPU verification
├── generate_summary_report.py   ★ benchmark and speedup report
├── snap_to_mtx.py               ★ SNAP → .mtx converter
└── edgelist_to_mtx.py           ★ edge list → .mtx converter
```

★ = added or written for this work. The paper's three GPU kernels and its CPU algorithm are **unmodified**.

---

## Known Issues and Open Work

**A pre-existing issue in the original code.** On genuinely directed graphs, the delta phase applies `A'` (predecessors) where the Brandes algorithm calls for `A` (successors). The CPU and all four GPU implementations share this behaviour, so they agree with each other but differ from `networkx`. Symmetric graphs are unaffected (`A = A'`), and no timing comparison is affected, since all five implementations compute the same quantity. This was not introduced by the tensor-core work; it is reported, not fixed.

**Tensor-core ablation not yet run.** The speedup comes from three sources — 16-root batching, the condensed layout, and the tensor cores themselves — and the third has not been isolated. The decisive experiment is the same layout and the same batching with an FMA loop on CUDA cores in place of the MMA. The expectation is that tensor cores do matter, because the condensed layout deliberately performs ~16x redundant arithmetic in exchange for coalesced memory access, and that trade only pays when arithmetic is nearly free. But that is a prediction, not a measurement.

**32-bit edge count.** `nz` is passed to the GPU entry points as `int`, capping all four algorithms at 2.15 billion edges. Widening it to `size_t` is a change to five signatures and would leave memory as the only limit. Not needed at the sizes studied here.

---

## Citing the Original Work

This repository builds on, and would not exist without:

> Oswaldo Artiles and Fahad Saeed. 2021. **TurboBC: A Memory Efficient and Scalable GPU Based Betweenness Centrality Algorithm in the Language of Linear Algebra.** In *50th International Conference on Parallel Processing Workshop (ICPP Workshops '21)*, August 9–12, 2021, Lemont, IL, USA. ACM, New York, NY, USA, 10 pages.
> https://doi.org/10.1145/3458744.3474047

Original code: https://github.com/pcdslab/TurboBC

---

## Acknowledgements

The original TurboBC research was supported by the National Science Foundation under Award Number CAREER OAC-1925960, and NVIDIA donated the K-40c Tesla and TITAN Xp GPUs used for the experiments in that paper. The content is solely the responsibility of its authors and does not necessarily represent the official views of the National Science Foundation.
