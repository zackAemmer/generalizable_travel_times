#!/bin/bash
set -e

RUN_NAME="full_run"


cd ~/Desktop/valle

# Run Models
python ./src/run_one_model.py "FF" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "CONV" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "GRU" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "TRSF" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "DEEP_TTE" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "FF_GTFS" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "CONV_GTFS" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "GRU_GTFS" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "TRSF_GTFS" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "DEEP_TTE_GTFS" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "FF_GRID" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "CONV_GRID" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "GRU_GRID" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "TRSF_GRID" "./results/$RUN_NAME/" "kcm/" False False

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
