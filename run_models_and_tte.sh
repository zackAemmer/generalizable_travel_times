#!/bin/zsh

# Move to project folder
cd ~/Desktop/valle

# Run setup
python ./src/prepare_run.py

# Run models
python ./src/run_models.py


## KCM
# Copy files to Deeptte
cp -a ./results/3_mo_cross_val/kcm/deeptte_formatted/ ../DeepTTE_Annotated/data
cp ../DeepTTE_Annotated/data/config.json ../DeepTTE_Annotated/config.json

# Run Deeptte
cd ~/Desktop/DeepTTE_Annotated
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
python main.py --task train --batch_size 10 --epochs 20 --result_file ./result/deeptte.res --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file train_log

# Copy Deeptte results back to results folder
cd ~/Desktop/valle
cp -a ../DeepTTE_Annotated/result ./results/3_mo_cross_val/kcm/deeptte_results/
cp -a ../DeepTTE_Annotated/saved_weights ./results/3_mo_cross_val/kcm/deeptte_results/


## ATB
# Copy files to Deeptte
cp -a ./results/3_mo_cross_val/atb/deeptte_formatted/ ../DeepTTE_Annotated/data
cp ../DeepTTE_Annotated/data/config.json ../DeepTTE_Annotated/config.json

# Run Deeptte
cd ~/Desktop/DeepTTE_Annotated
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
python main.py --task train --batch_size 10 --epochs 20 --result_file ./result/deeptte.res --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file train_log

# Copy Deeptte results back to results folder
cd ~/Desktop/valle
cp -a ../DeepTTE_Annotated/result ./results/3_mo_cross_val/atb/deeptte_results/
cp -a ../DeepTTE_Annotated/saved_weights ./results/3_mo_cross_val/atb/deeptte_results/