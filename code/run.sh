#!/bin/bash

cd ../code/

SEED_LIST=($(seq 30 31))
max_jobs=1

START=6
END=6
Device="cuda:0"
check_jobs() {
    while [ $(jobs -rp | wc -l) -ge $max_jobs ]; do
        sleep 1
    done
}

for SEED in "${SEED_LIST[@]}"; do
    for i in $(seq $START $END); do
        check_jobs
        echo "Running training with top $i variables with SEED=$SEED" | tee -a $log_file
        python main_Pipeline.py --SEED $SEED --range_val $i --device $Device
    done
done

wai