#!/bin/bash
set -e

RUN_NAME="debug"


cd ~/Skrivebord/valle
# cd ~/Desktop/valle


# python ./src/run_preparation.py


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

python ./src/run_one_model.py "GRU_GRID" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "TRSF_GRID" "./results/$RUN_NAME/" "kcm/" False False


# python ./src/run_experiments.py