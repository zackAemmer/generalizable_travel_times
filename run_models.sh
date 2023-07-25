#!/bin/bash
set -e


# cd ~/Skrivebord/valle
cd ~/Desktop/valle

# python ./src/run_preparation.py
python ./src/run_one_model.py "FF" "./results/full_run/" "kcm/"
python ./src/run_one_model.py "FF_GRID" "./results/full_run/" "kcm/"
python ./src/run_one_model.py "CONV" "./results/full_run/" "kcm/"
python ./src/run_one_model.py "CONV_GRID" "./results/full_run/" "kcm/"
python ./src/run_one_model.py "GRU" "./results/full_run/" "kcm/"
python ./src/run_one_model.py "GRU_GRID" "./results/full_run/" "kcm/"
python ./src/run_one_model.py "TRSF" "./results/full_run/" "kcm/"
python ./src/run_one_model.py "TRSF_GRID" "./results/full_run/" "kcm/"
python ./src/run_one_model.py "DEEP_TTE" "./results/full_run/" "kcm/"

python ./src/run_one_model.py "FF" "./results/full_run/" "atb/"
python ./src/run_one_model.py "FF_GRID" "./results/full_run/" "atb/"
python ./src/run_one_model.py "CONV" "./results/full_run/" "atb/"
python ./src/run_one_model.py "CONV_GRID" "./results/full_run/" "atb/"
python ./src/run_one_model.py "GRU" "./results/full_run/" "atb/"
python ./src/run_one_model.py "GRU_GRID" "./results/full_run/" "atb/"
python ./src/run_one_model.py "TRSF" "./results/full_run/" "atb/"
python ./src/run_one_model.py "TRSF_GRID" "./results/full_run/" "atb/"
python ./src/run_one_model.py "DEEP_TTE" "./results/full_run/" "atb/"

# python ./src/run_experiments.py