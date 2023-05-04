#!/bin/zsh


### Move to project folder ###
cd ~/Skrivebord/valle


### KCM ###
# Copy files to Deeptte (test on AtB)
cp -a ./results/debug/atb/deeptte_formatted/* ../DeepTTE/data
cp ../DeepTTE/data/config.json ../DeepTTE/config.json
# Run Deeptte
cd ~/Skrivebord/DeepTTE
rm -rf ./result && mkdir ./result
rm -rf ./saved_weights && mkdir ./saved_weights
rm -rf ./logs && mkdir ./logs
cp ../valle/results/debug/kcm/deeptte_results/weights_4 ./saved_weights/weights_4
python main.py --task test --batch_size 10 --epochs 50 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file test_log --weight_file ./saved_weights/weights_4 
# Copy Deeptte results back to results folder
cd ~/Skrivebord/valle
cp -a ../DeepTTE/result ./results/debug/kcm/deeptte_gen_results/


# ### ATB ###
# # Copy files to Deeptte (test on KCM)
# cp -a ./results/debug/kcm/deeptte_formatted/* ../DeepTTE/data
# cp ../DeepTTE/data/config.json ../DeepTTE/config.json
# # Run Deeptte
# cd ~/Skrivebord/DeepTTE
# rm -rf ./result && mkdir ./result
# rm -rf ./saved_weights && mkdir ./saved_weights
# rm -rf ./logs && mkdir ./logs
# python main.py --task train --batch_size 10 --epochs 50 --result_file ./result/deeptte --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file train_log
# # Copy Deeptte results back to results folder
# cd ~/Skrivebord/valle
# cp -a ../DeepTTE/result ./results/debug/atb/deeptte_results/
# cp -a ../DeepTTE/saved_weights ./results/debug/atb/deeptte_results/