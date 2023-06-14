#!/bin/bash
set -e
run_name=debug
n_folds=3
epochs=30


### Move to project folder ###
cd ~/Skrivebord/valle

### KCM ###
# Copy files to Deeptte
cp -a ./results/$run_name/kcm/deeptte_formatted/* ../DeepTTE/data
cp ../DeepTTE/data/train_config.json ../DeepTTE/config.json
# Run Deeptte
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
python main.py --task train --batch_size 10 --epochs $epochs --n_folds $n_folds --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file train_log
# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
cp -a ../DeepTTE/result ./results/$run_name/kcm/deeptte_results/
cp -a ../DeepTTE/saved_weights ./results/$run_name/kcm/deeptte_results/

### ATB ###
# Copy files to Deeptte
cp -a ./results/$run_name/atb/deeptte_formatted/* ../DeepTTE/data
cp ../DeepTTE/data/train_config.json ../DeepTTE/config.json
# Run Deeptte
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
python main.py --task train --batch_size 10 --epochs $epochs --n_folds $n_folds --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file train_log
# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
cp -a ../DeepTTE/result ./results/$run_name/atb/deeptte_results/
cp -a ../DeepTTE/saved_weights ./results/$run_name/atb/deeptte_results/