#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/../graphData"
LOGFILE="$HERE/bc_verification_results.log"

rm -f "$LOGFILE"
echo "=================================================================================" >> "$LOGFILE"
echo "               TurboBC Execution & CPU vs GPU Verification Log                   " >> "$LOGFILE"
echo "               System: NVIDIA L4 GPU | Architecture: sm_89                       " >> "$LOGFILE"
echo "=================================================================================" >> "$LOGFILE"
echo "" >> "$LOGFILE"

echo "================================================================================="
echo "  🚀 RUNNING TURBOBC BENCHMARK SUITE (CPU & NVIDIA L4 GPU)                       "
echo "  Note: All vertex BC scores are stored & cross-checked directly in log files!  "
echo "================================================================================="

COUNT=1

run_test() {
    file=$1
    w=$2
    ug=$3
    format=$4
    p=$5
    nr=$6
    rs=$7
    repet=$8
    seq=$9

    if [ "$format" -eq "0" ]; then
        fmt_name="scCSC"
    elif [ "$format" -eq "1" ]; then
        fmt_name="vcCSC"
    elif [ "$format" -eq "2" ]; then
        fmt_name="scCOOC"
    else
        fmt_name="tcTF32"
    fi

    printf "  [%2d/48] Computing & Storing BC Scores for %-16s | Algo: %-6s ... " "$COUNT" "$file.mtx" "$fmt_name"

    echo "---------------------------------------------------------------------------------" >> "$LOGFILE"
    echo "[EXPERIMENT] Graph: $file.mtx | Format: $fmt_name" >> "$LOGFILE"
    echo "[COMMAND] ./exec $DATA/$file.mtx --w $w --ug $ug --format $format --p $p --nr $nr --rs $rs --repet $repet --seq $seq" >> "$LOGFILE"
    echo "---------------------------------------------------------------------------------" >> "$LOGFILE"
    ./exec "$DATA/$file.mtx" "$w" "$ug" "$format" "$p" "$nr" "$rs" "$repet" "$seq" >> "$LOGFILE" 2>&1
    echo "" >> "$LOGFILE"
    
    echo "✅ Done (Stored in File)"
    ((COUNT++))
}

# Run all 4 algorithms for each of the 12 test graphs
for file in mycielskian4; do
    run_test $file 0 1 0 1 1 0 100 1
    run_test $file 0 1 1 1 1 0 100 1
    run_test $file 0 1 2 1 1 0 100 1
    run_test $file 0 1 3 1 1 0 100 1
done

for file in Tina_AskCal; do
    run_test $file 0 0 0 1 1 10 1 1
    run_test $file 0 0 1 1 1 10 1 1
    run_test $file 0 0 2 1 1 10 1 1
    run_test $file 0 0 3 1 1 10 1 1
done

for file in Ragusa18 cage4; do
    run_test $file 1 0 0 1 1 0 10 1
    run_test $file 1 0 1 1 1 0 10 1
    run_test $file 1 0 2 1 1 0 10 1
    run_test $file 1 0 3 1 1 0 10 1
done

# From here down every graph runs nr=32 roots.  tcTF32 always computes a full
# TCK=16-wide batch, so timing it at nr=1 charges it for 16 roots and credits
# it with one; nr must be a multiple of 16 for the four algorithms to be
# compared on equal work.  The CUDA-core kernels are single-source, so their
# per-root cost is unchanged by nr.
for file in smallworld; do
    run_test $file 0 1 0 1 32 0 10 1
    run_test $file 0 1 1 1 32 0 10 1
    run_test $file 0 1 2 1 32 0 10 1
    run_test $file 0 1 3 1 32 0 10 1
done

for file in ASIC_100ks; do
    run_test $file 1 0 0 1 32 0 10 1
    run_test $file 1 0 1 1 32 0 10 1
    run_test $file 1 0 2 1 32 0 10 1
    run_test $file 1 0 3 1 32 0 10 1
done

# congress: directed (ug 0) and carries a value column (w 1); converted from
# congress.edgelist by edgelist_to_mtx.py
for file in congress; do
    run_test $file 1 0 0 1 32 0 10 1
    run_test $file 1 0 1 1 32 0 10 1
    run_test $file 1 0 2 1 32 0 10 1
    run_test $file 1 0 3 1 32 0 10 1
done

# SNAP real-world social networks, converted by snap_to_mtx.py.  All three are
# pattern .mtx with both edge directions written out, hence w 0 / ug 0.  These
# are the sparsest graphs in the suite (0.4%-0.8% density under a geometric
# 16x16 tiling) and are what the condensed blocking in bcgputc.cu was built for.
for file in fb_new_sites fb_artist epinions; do
    run_test $file 0 0 0 1 32 0 10 1
    run_test $file 0 0 1 1 32 0 10 1
    run_test $file 0 0 2 1 32 0 10 1
    run_test $file 0 0 3 1 32 0 10 1
done

# nd3k / nd6k: 3D finite-element meshes from the SuiteSparse ND set, the
# naturally dense end of the suite -- 60% chunk occupancy against 6%-15% for
# the social and circuit graphs. Symmetric already, so ug 0.
for file in nd3k nd6k; do
    run_test $file 0 0 0 1 32 0 10 1
    run_test $file 0 0 1 1 32 0 10 1
    run_test $file 0 0 2 1 32 0 10 1
    run_test $file 0 0 3 1 32 0 10 1
done

echo "================================================================================="
echo "All 48 tests completed! Complete vertex logs saved to: $LOGFILE"
echo "================================================================================="

# Step 1: Extract CPU Baseline & Cross-Check GPU Scores
python3 "$HERE/cross_check_scores.py"

# Step 2: Display Benchmark and Speedup Summary Report
python3 "$HERE/generate_summary_report.py" "$LOGFILE"
