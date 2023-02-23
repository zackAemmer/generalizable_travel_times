#!/bin/zsh

# Move to project folder
cd ~/Desktop/valle
# cd ~/Skrivebord/valle

# Run setup
python ./src/prepare_run.py

# Run models
python ./src/run_models.py