#!/usr/bin/env bash
TEST="1"
DATA="/home/akshat/TurboBC/graphData"

for file in mycielskian4 
do
    if [ "$TEST" = "1" ] ; then
	echo $file CSC
	echo cuBC_CSC_COOC   ./exec $DATA/$file.mtx --w 0 --ug 1 --format  0 --p 1 --nr 1 --rs 0 --repet 100 --seq 1
	./exec $DATA/$file.mtx  0 1 0 1 1 0 100 1
	echo $file COOC
	echo cuBC_CSC_COOC   ./exec $DATA/$file.mtx --w 0 --ug 1 --format  2 --p 1 --nr 1 --rs 0 --repet 100 --seq 1
	./exec $DATA/$file.mtx  0 1 2 1 1 0 100 1
    fi
done

for file in Tina_AskCal
do
    if [ "$TEST" = "1" ] ; then
	echo $file CSC
	echo cuBC_CSC_COOC ./exec $DATA/$file.mtx --w 0 --ug  0 --format  0 --p 1 --nr 1 --rs 10 --repet 1 --seq 1
	./exec $DATA/$file.mtx  0 0 0 1 1 10 1 1
	echo $file COOC
	echo cuBC_CSC_COOC ./exec $DATA/$file.mtx --w 0 --ug  0 --format  2 --p 1 --nr 1 --rs 10 --repet 1 --seq 1
	./exec $DATA/$file.mtx  0 0 2 1 1 10 1 1
    fi
done

for file in Ragusa18 cage4
do
    if [ "$TEST" = "1" ] ; then
	echo $file CSC
	echo cuBC_CSC_COOC ./exec $DATA/$file.mtx --w 1 --ug  0 --format  0 --p 1   --nr 1 --rs 0 --repet 10 --seq 1
	./exec $DATA/$file.mtx  1 0 0 1 1 0 10 1
	echo $file COOC
	echo cuBC_CSC_COOC ./exec $DATA/$file.mtx --w 1 --ug  0 --format  2 --p 1   --nr 1 --rs 0 --repet 10 --seq 1
	./exec $DATA/$file.mtx  1 0 2 1 1 0 10 1
    fi
done

for file in smallworld 
do
    if [ "$TEST" = "1" ] ; then
	echo $file CSC
	echo cuBC_CSC_COOC ./exec $DATA/$file.mtx --w 0 --ug 1 --format  0 --p 1 --nr 1 --rs 0 --repet 100 --seq 1
        ./exec $DATA/$file.mtx  0 1 0 1 1 0 100  1
	echo $file COOC
	echo cuBC_CSC_COOC ./exec $DATA/$file.mtx --w 0 --ug 1 --format  2 --p 1 --nr 1 --rs 0 --repet 100 --seq 1
	./exec $DATA/$file.mtx  0 1 2 1 1 0 100  1
    fi
done

for file in  ASIC_100ks 
do
    if [ "$TEST" = "1" ] ; then
	echo $file CSC
	echo cuBC_CSC_COOC ./exec $DATA/$file.mtx --w 1 --ug 0 --format  0 --p 1 --nr 1 --rs 100 --repet 100 --seq 1
	./exec $DATA/$file.mtx  1 0 0 1 1 100 100  1
	echo $file COOC
	echo cuBC_CSC_COOC ./exec $DATA/$file.mtx --w 1 --ug 0 --format  2 --p 1 --nr 1 --rs 100 --repet 100 --seq 1
	./exec $DATA/$file.mtx  1 0 2 1 1 100 100  1
    fi
done


