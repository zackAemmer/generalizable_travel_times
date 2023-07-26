#!/bin/bash
set -e

RUN_NAME="debug_nosch"


cd ~/Skrivebord/valle
# cd ~/Desktop/valle


# python ./src/run_preparation.py


python ./src/run_one_model.py "FF" "./results/$RUN_NAME/" "kcm_atb/" True False
python ./src/run_one_model.py "CONV" "./results/$RUN_NAME/" "kcm_atb/" True False
python ./src/run_one_model.py "GRU" "./results/$RUN_NAME/" "kcm_atb/" True False
python ./src/run_one_model.py "TRSF" "./results/$RUN_NAME/" "kcm_atb/" True False
python ./src/run_one_model.py "DEEP_TTE" "./results/$RUN_NAME/" "kcm_atb/" True False


# python ./src/run_experiments.py