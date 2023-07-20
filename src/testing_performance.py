#!/usr/bin python3


import gc
import itertools
import json
import os
import random
import time
import timeit
import h5py

import numpy as np
import torch
import pyarrow as pa
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn import metrics
from tabulate import tabulate
import pyarrow.dataset as ds

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
        PIN_MEMORY = True
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
        NUM_WORKERS = 0
        PIN_MEMORY = False

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
    with open(f"{run_folder}{train_network_folder}deeptte_formatted/train_summary_config.json", "r") as f:
        config = json.load(f)





    dataset_new = data_loader.BetterGenericDataset(f"{run_folder}{train_network_folder}deeptte_formatted/train", config, holdout_routes=kwargs['holdout_routes'])
    print(len(dataset_new))

    t0 = time.time()
    z = dataset_new.get_all_samples(keep_cols=["shingle_id","locationtime","x","y"])
    time_iter = time.time() - t0
    print(f"{time_iter} seconds")

    t0 = time.time()
    for i in range(0,1000):
        z = dataset_new.__getitem__(i)
    time_iter = time.time() - t0
    print(f"{time_iter*len(dataset_new)/1000/60} minutes")

    loader = DataLoader(dataset_new, batch_size=512, collate_fn=data_loader.basic_collate, pin_memory=True, num_workers=4, drop_last=True)
    t0 = time.time()
    i=0
    for x in loader:
        if i==100:
            break
        i+=1
        z = x
    time_iter = time.time() - t0
    print(f"{time_iter*len(loader)/100/60} minutes")






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
        run_folder="./results/param_search/",
        train_network_folder="kcm/",
        test_network_folder="atb/",
        tune_network_folder="atb/",
        TUNE_EPOCHS=2,
        BATCH_SIZE=32,
        LEARN_RATE=1e-3,
        HIDDEN_SIZE=32,
        grid_s_size=500,
        data_subset=.1,
        n_tune_samples=100,
        n_folds=2,
        holdout_routes=[100252,100139,102581,100341,102720]
    )