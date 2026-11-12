# results/

`turbobc_results_20260926_1546.txt` is the terminal output of the full
benchmark run that the HiPC paper's numbers come from (52 tests: 13 graphs x
4 kernels, NVIDIA L4, 32 sources, 10 repetitions and 3 on amazon). Table I
and Fig. 1 of the paper use the smallworld, fb_artist, nd6k and amazon rows of
its summary table.

It was produced by

```bash
cd TurboBC && ./run_all_experiments.sh 2>&1 | tee ../results/turbobc_results_$(date +%Y%m%d_%H%M).txt
```

The per-vertex log of the same run (`TurboBC/bc_verification_results.log`,
about 230 MB) is too large for GitHub and is not committed.
