#!/bin/bash
set -e

# RUN_NAME="debug_nosch"
RUN_NAME="full_run_nosch"


cd ~/Skrivebord/valle
# cd ~/Desktop/valle


# Run Models
python ./src/run_one_model.py "FF" "./results/$RUN_NAME/" "kcm_atb/" True False
python ./src/run_one_model.py "CONV" "./results/$RUN_NAME/" "kcm_atb/" True False
python ./src/run_one_model.py "GRU" "./results/$RUN_NAME/" "kcm_atb/" True False
python ./src/run_one_model.py "TRSF" "./results/$RUN_NAME/" "kcm_atb/" True False
python ./src/run_one_model.py "DEEP_TTE" "./results/$RUN_NAME/" "kcm_atb/" True False

# Run Experiments
python ./src/run_experiments.py "FF" "./results/$RUN_NAME/" "kcm_atb/" "rut/" "rut/" True False
python ./src/run_experiments.py "CONV" "./results/$RUN_NAME/" "kcm_atb/" "rut/" "rut/" True False
python ./src/run_experiments.py "GRU" "./results/$RUN_NAME/" "kcm_atb/" "rut/" "rut/" True False
python ./src/run_experiments.py "TRSF" "./results/$RUN_NAME/" "kcm_atb/" "rut/" "rut/" True False
python ./src/run_experiments.py "DEEP_TTE" "./results/$RUN_NAME/" "kcm_atb/" "rut/" "rut/" True False
