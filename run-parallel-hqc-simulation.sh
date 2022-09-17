#!/bin/bash

set -e 
set -o pipefail

live=0

for I in {0..99}
do 
    for W in 10 20 30 40 50
    do
        for E in 0.10 0.05 0.005 0.0
        do 
            LOGFILE="simulation-data/hqc-simulation-E$E-W$W-$I.log"
            CMD="simulate-with-python/main.py hqc_simulate \
                --verbose \
                --decode-every 100 \
                --key-file test-hqc.key \
                --csv-output simulation-data/hqc-simulation-E$E-W$W.csv \
                --code-weight $W \
                --label $I \
                --error-rate $E"
            if [[ ! -f "$LOGFILE" ]];
            then
                echo "Launching: $CMD"
                ($CMD 2>&1 | tee $LOGFILE | grep -v -E 'DEBUG|INFO') &
                let live=live+1
                if [[ $live > 4 ]];
                then
                    # wait for first job to return so we can start the next
                    wait -n -p id || { echo 'Command failed, aborting script'; exit 1; }
                    let live=live-1
                    echo "$id finished ($live remaining)"
                fi
            else
                echo "Skipping $CMD"
            fi
        done
    done
done

echo "waiting for $live remaining jobs"
wait
echo "Done"