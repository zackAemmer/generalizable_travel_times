#!/bin/bash

run_name=medium

### Move to project folder ###
cd ~/Skrivebord/valle

### KCM on KCM ###
cp -a ./results/$run_name/kcm/deeptte_formatted/* ../DeepTTE/data
cp ./results/$run_name/kcm/deeptte_formatted/train_config.json ../DeepTTE/config.json
# Run Deeptte on validation set for each network
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
cp ../valle/results/$run_name/kcm/deeptte_results/saved_weights/weights_0 ./saved_weights/weights_0
python main.py --task test --batch_size 10 --epochs 2 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file test_log --weight_file ./saved_weights/weights_0 --flag KCM_KCM
# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
mkdir -p ./results/$run_name/kcm/deeptte_results/generalization/
cp -r ../DeepTTE/result/* ./results/$run_name/kcm/deeptte_results/generalization/

### KCM on ATB ###
cp -a ./results/$run_name/atb/deeptte_formatted/* ../DeepTTE/data
cp ./results/$run_name/kcm/deeptte_formatted/train_config.json ../DeepTTE/config.json
# Run Deeptte on validation set for each network
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
cp ../valle/results/$run_name/kcm/deeptte_results/saved_weights/weights_0 ./saved_weights/weights_0
python main.py --task test --batch_size 10 --epochs 2 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file test_log --weight_file ./saved_weights/weights_0 --flag KCM_ATB
# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
mkdir -p ./results/$run_name/kcm/deeptte_results/generalization/
cp -r ../DeepTTE/result/* ./results/$run_name/kcm/deeptte_results/generalization/

### ATB on ATB ###
cp -a ./results/$run_name/atb/deeptte_formatted/* ../DeepTTE/data
cp ./results/$run_name/atb/deeptte_formatted/train_config.json ../DeepTTE/config.json
# Run Deeptte on validation set for each network
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
cp ../valle/results/$run_name/atb/deeptte_results/saved_weights/weights_0 ./saved_weights/weights_0
python main.py --task test --batch_size 10 --epochs 2 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file test_log --weight_file ./saved_weights/weights_0 --flag ATB_ATB
# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
mkdir -p ./results/$run_name/atb/deeptte_results/generalization/
cp -r ../DeepTTE/result/* ./results/$run_name/atb/deeptte_results/generalization/

### ATB on KCM ###
cp -a ./results/$run_name/kcm/deeptte_formatted/* ../DeepTTE/data
cp ./results/$run_name/atb/deeptte_formatted/train_config.json ../DeepTTE/config.json
# Run Deeptte on validation set for each network
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
cp ../valle/results/$run_name/atb/deeptte_results/saved_weights/weights_0 ./saved_weights/weights_0
python main.py --task test --batch_size 10 --epochs 2 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file test_log --weight_file ./saved_weights/weights_0 --flag ATB_KCM
# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
mkdir -p ./results/$run_name/atb/deeptte_results/generalization/
cp -r ../DeepTTE/result/* ./results/$run_name/atb/deeptte_results/generalization/