#!/bin/zsh

### Move to project folder ###
cd ~/Skrivebord/valle

### KCM on KCM ###
cp -a ./results/debug/kcm/deeptte_formatted/* ../DeepTTE/data
cp ./results/debug/kcm/deeptte_formatted/config.json ../DeepTTE/config.json
# Run Deeptte on validation set for each network
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
cp ../valle/results/debug/kcm/deeptte_results/saved_weights/weights_4 ./saved_weights/weights_4
python main.py --task test --batch_size 10 --epochs 2 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file test_log --weight_file ./saved_weights/weights_4 --flag KCM_KCM
# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
cp -a ../DeepTTE/result ./results/debug/kcm/deeptte_results/generalization

### KCM on ATB ###
cp -a ./results/debug/atb/deeptte_formatted/* ../DeepTTE/data
cp ./results/debug/kcm/deeptte_formatted/config.json ../DeepTTE/config.json
# Run Deeptte on validation set for each network
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
cp ../valle/results/debug/kcm/deeptte_results/saved_weights/weights_4 ./saved_weights/weights_4
python main.py --task test --batch_size 10 --epochs 2 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file test_log --weight_file ./saved_weights/weights_4 --flag KCM_ATB
# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
cp -a ../DeepTTE/result ./results/debug/kcm/deeptte_results/generalization

### ATB on ATB ###
cp -a ./results/debug/atb/deeptte_formatted/* ../DeepTTE/data
cp ./results/debug/atb/deeptte_formatted/config.json ../DeepTTE/config.json
# Run Deeptte on validation set for each network
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
cp ../valle/results/debug/atb/deeptte_results/saved_weights/weights_4 ./saved_weights/weights_4
python main.py --task test --batch_size 10 --epochs 2 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file test_log --weight_file ./saved_weights/weights_4 --flag ATB_ATB
# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
cp -a ../DeepTTE/result ./results/debug/atb/deeptte_results/generalization

### ATB on KCM ###
cp -a ./results/debug/kcm/deeptte_formatted/* ../DeepTTE/data
cp ./results/debug/atb/deeptte_formatted/config.json ../DeepTTE/config.json
# Run Deeptte on validation set for each network
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
cp ../valle/results/debug/atb/deeptte_results/saved_weights/weights_4 ./saved_weights/weights_4
python main.py --task test --batch_size 10 --epochs 2 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file test_log --weight_file ./saved_weights/weights_4 --flag ATB_KCM
# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
cp -a ../DeepTTE/result ./results/debug/atb/deeptte_results/generalization