#!/bin/bash
set -e
run_name=debug
n_folds=3
epochs=10


### Move data to folder ###
cd ~/Skrivebord/valle
cp -a ./results/$run_name/kcm/deeptte_formatted/* ../DeepTTE/data/kcm/
cp -a ./results/$run_name/atb/deeptte_formatted/* ../DeepTTE/data/atb/

### KCM ###
cp ../DeepTTE/data/kcm/train_config.json ../DeepTTE/config.json
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
cp ../valle/results/$run_name/kcm/deeptte_results/saved_weights/* ./saved_weights/
python main.py --task generalize --train_network kcm --test_network atb --batch_size 10 --epochs 10 --n_folds $n_folds --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file generalize_log
# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
mkdir -p ./results/$run_name/kcm/deeptte_results/generalization/
cp -r ../DeepTTE/result/* ./results/$run_name/kcm/deeptte_results/generalization/

### AtB ###
cp ../DeepTTE/data/atb/train_config.json ../DeepTTE/config.json
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
cp ../valle/results/$run_name/atb/deeptte_results/saved_weights/* ./saved_weights/
python main.py --task generalize --train_network atb --test_network kcm --batch_size 10 --epochs 10 --n_folds $n_folds --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file generalize_log
# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
mkdir -p ./results/$run_name/atb/deeptte_results/generalization/
cp -r ../DeepTTE/result/* ./results/$run_name/atb/deeptte_results/generalization/