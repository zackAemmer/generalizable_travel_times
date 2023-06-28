#!/usr/bin python3


import gc
import itertools
import json
import os
import random
import time

import numpy as np
import torch
from sklearn import metrics
from tabulate import tabulate

from models import avg_speed, conv, ff, grids, persistent, rnn, schedule, transformer
from utils import data_loader, data_utils, model_utils

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

import os.path as path

def run(run_folder, train_network_folder, **kwargs):
    # with profile(activities=[ProfilerActivity.CPU]) as prof:

    print("="*30)

    # Select device to train on, and number workers if GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        NUM_WORKERS = 8
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
        NUM_WORKERS = 0
    print(f"DEVICE: {device}")
    print(f"WORKERS: {NUM_WORKERS}")

    # Define embedded variables for network models
    embed_dict = {
        'timeID': {
            'vocab_size': 1440,
            'embed_dims': 27
        },
        'weekID': {
            'vocab_size': 7,
            'embed_dims': 4
        }
    }

    # Data loading and fold setup
    print(f"DATA: '{run_folder}{train_network_folder}deeptte_formatted/'")
    with open(f"{run_folder}{train_network_folder}deeptte_formatted/train_config.json", "r") as f:
        config = json.load(f)
    dataset = data_loader.GenericDataset(f"{run_folder}{train_network_folder}deeptte_formatted/train", config, holdout_routes=kwargs['holdout_routes'])
    grid = grids.NGridBetter(config['grid_bounds'], kwargs['grid_s_size'])
    grid.add_grid_content(data_utils.map_from_deeptte(dataset.content,["locationtime","x","y","speed_m_s","bearing"]))
    grid.build_cell_lookup()
    dataset.grid = grid

    # Prefetch
    # reverse sort
    # cache time values
    
    t0 = time.time()
    dataset.add_grid_features = False
    for batch in dataset:
        x = len(batch['lngs'])
    train_time = (time.time() - t0)
    print(train_time)

    t0 = time.time()
    dataset.add_grid_features = True
    for batch in dataset:
        x = len(batch['lngs'])
    train_time = (time.time() - t0)
    print(train_time)





    # prof.export_chrome_trace("trace.json")
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))


        # full_set = 2300000*5
        # data = np.ones(full_set*12*64,dtype="float32").reshape(full_set,12,64)
        # data_utils.write_pkl(data)
        # filename = path.join('newfile.dat')
        # fp = np.memmap(filename, dtype='object', mode='w+', shape=(100000,12))
        # fp[:] = data[:]
        # fp.flush()
        # newfp = np.memmap(filename, dtype='object', mode='r', shape=data.shape)

if __name__=="__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run(
        run_folder="./results/debug/",
        train_network_folder="kcm/",
        test_network_folder="atb/",
        tune_network_folder="atb/",
        TUNE_EPOCHS=10,
        BATCH_SIZE=64,
        LEARN_RATE=1e-3,
        HIDDEN_SIZE=32,
        grid_s_size=500,
        data_subset=.1,
        n_tune_samples=100,
        n_folds=3,
        holdout_routes=[100252,100139,102581,100341,102720]
    )