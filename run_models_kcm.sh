#!/bin/bash
set -e


cd ~/Skrivebord/valle
# cd ~/Desktop/valle


# python ./src/run_preparation.py


RUN_NAME="full_run"
# python ./src/run_one_model.py "FF" "./results/$RUN_NAME/" "kcm/" False False
# python ./src/run_one_model.py "FF_GRID" "./results/$RUN_NAME/" "kcm/" False False
# python ./src/run_one_model.py "CONV" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "CONV_GRID" "./results/$RUN_NAME/" "kcm/" False False
# python ./src/run_one_model.py "GRU" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "GRU_GRID" "./results/$RUN_NAME/" "kcm/" False False
# python ./src/run_one_model.py "TRSF" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "TRSF_GRID" "./results/$RUN_NAME/" "kcm/" False False
python ./src/run_one_model.py "DEEP_TTE" "./results/$RUN_NAME/" "kcm/" False False


# python ./src/run_experiments.py