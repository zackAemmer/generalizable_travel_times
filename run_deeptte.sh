#!/bin/zsh

# Move to project folder
# cd ~/Desktop/valle
cd ~/Skrivebord/valle


## KCM
# Copy files to Deeptte
cp -a ./results/3_mo_cross_val/kcm/deeptte_formatted/ ../DeepTTE/data
cp ../DeepTTE/data/config.json ../DeepTTE/config.json

# Run Deeptte
# cd ~/Desktop/DeepTTE
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
python main.py --task train --batch_size 10 --epochs 50 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file train_log

# Copy Deeptte results back to results folder
# cd ~/Desktop/valle
cd ~/Skrivebord/valle
cp -a ../DeepTTE/result ./results/3_mo_cross_val/kcm/deeptte_results/
cp -a ../DeepTTE/saved_weights ./results/3_mo_cross_val/kcm/deeptte_results/


## ATB
# Copy files to Deeptte
cp -a ./results/3_mo_cross_val/atb/deeptte_formatted/ ../DeepTTE/data
cp ../DeepTTE/data/config.json ../DeepTTE_Annotated/config.json

# Run Deeptte
# cd ~/Desktop/DeepTTE
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
python main.py --task train --batch_size 10 --epochs 50 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file train_log

# Copy Deeptte results back to results folder
cd ~/Desktop/valle
cd ~/Skrivebord/valle
cp -a ../DeepTTE/result ./results/3_mo_cross_val/atb/deeptte_results/
cp -a ../DeepTTE/saved_weights ./results/3_mo_cross_val/atb/deeptte_results/