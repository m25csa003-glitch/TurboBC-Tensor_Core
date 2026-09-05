#!/usr/bin/env bash
# BC on the congress graph (475 vertices, 13289 directed weighted edges).
# The graph is directed and asymmetric, so --ug 0; the .mtx carries a value
# column, so --w 1.
DATA="/home/akshat/TurboBC/graphData"
NR=475          # number of root vertices; 475 = every vertex = full BC
REPET=1

for file in congress
do
    echo "$file CSC_sc"
    echo cuBC_CSC_COOC ./exec $DATA/$file.mtx --w 1 --ug 0 --format 0 --p 1 --nr $NR --rs 0 --repet $REPET --seq 1
    ./exec $DATA/$file.mtx  1 0 0 1 $NR 0 $REPET 1

    echo "$file CSC_wa"
    echo cuBC_CSC_COOC ./exec $DATA/$file.mtx --w 1 --ug 0 --format 1 --p 1 --nr $NR --rs 0 --repet $REPET --seq 1
    ./exec $DATA/$file.mtx  1 0 1 1 $NR 0 $REPET 1

    echo "$file COOC_sc"
    echo cuBC_CSC_COOC ./exec $DATA/$file.mtx --w 1 --ug 0 --format 2 --p 1 --nr $NR --rs 0 --repet $REPET --seq 1
    ./exec $DATA/$file.mtx  1 0 2 1 $NR 0 $REPET 1
done
