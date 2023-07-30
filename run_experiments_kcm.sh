#!/bin/bash
set -e

RUN_NAME="debug"
# RUN_NAME="full_run"


# cd ~/Skrivebord/valle
cd ~/Desktop/valle


# Run Experiments
python ./src/run_experiments.py "FF" "./results/$RUN_NAME/" "kcm/" "atb/" "atb/" False False
python ./src/run_experiments.py "CONV" "./results/$RUN_NAME/" "kcm/" "atb/" "atb/" False False
python ./src/run_experiments.py "GRU" "./results/$RUN_NAME/" "kcm/" "atb/" "atb/" False False
python ./src/run_experiments.py "TRSF" "./results/$RUN_NAME/" "kcm/" "atb/" "atb/" False False
python ./src/run_experiments.py "DEEP_TTE" "./results/$RUN_NAME/" "kcm/" "atb/" "atb/" False False
python ./src/run_experiments.py "FF_GTFS" "./results/$RUN_NAME/" "kcm/" "atb/" "atb/" False False
python ./src/run_experiments.py "CONV_GTFS" "./results/$RUN_NAME/" "kcm/" "atb/" "atb/" False False
python ./src/run_experiments.py "GRU_GTFS" "./results/$RUN_NAME/" "kcm/" "atb/" "atb/" False False
python ./src/run_experiments.py "TRSF_GTFS" "./results/$RUN_NAME/" "kcm/" "atb/" "atb/" False False
python ./src/run_experiments.py "DEEP_TTE_GTFS" "./results/$RUN_NAME/" "kcm/" "atb/" "atb/" False False
