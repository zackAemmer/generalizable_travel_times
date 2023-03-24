#!/bin/zsh

### Move to project folder ###
cd ~/Skrivebord/valle


### KCM ###
# Copy files to Deeptte
cp -a ./results/small/kcm/deeptte_formatted/* ../DeepTTE/data
cp ../DeepTTE/data/config.json ../DeepTTE/config.json

# Run Deeptte
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
python main.py --task train --batch_size 10 --epochs 50 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file train_log

# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
cp -a ../DeepTTE/result ./results/small/kcm/deeptte_results/
cp -a ../DeepTTE/saved_weights ./results/small/kcm/deeptte_results/


### ATB ###
# Copy files to Deeptte
cp -a ./results/small/atb/deeptte_formatted/* ../DeepTTE/data
cp ../DeepTTE/data/config.json ../DeepTTE/config.json

# Run Deeptte
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
python main.py --task train --batch_size 10 --epochs 50 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file train_log

# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
cp -a ../DeepTTE/result ./results/small/atb/deeptte_results/
cp -a ../DeepTTE/saved_weights ./results/small/atb/deeptte_results/