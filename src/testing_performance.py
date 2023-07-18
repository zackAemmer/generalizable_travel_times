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

    # dataset = data_loader.GenericDataset(f"{run_folder}{train_network_folder}deeptte_formatted/train", config, subset=0.5, holdout_routes=kwargs['holdout_routes'])
    # t0 = time.time()
    # for i in range(0,1000):
    #     dataset.__getitem__(i)
    # print(time.time()-t0)

    # # Assuming your data is stored in a NumPy array called 'data'
    # data = dataset_new.t.values
    # with h5py.File('data.h5', 'w') as f:
    #     f.create_dataset('tabular_data', data=dataset_new.t[['x_cent','y_cent']].values.astype(float))

    # with h5py.File('data.h5', 'r') as f:
    #     dataset = f['tabular_data']
    #     specific_lines = dataset[100:1000]

    dataset_new = data_loader.BetterGenericDataset(f"{run_folder}{train_network_folder}deeptte_formatted/train", config, subset=0.5, holdout_routes=kwargs['holdout_routes'])
    dataset_new.get_grid_samples()
    t0 = time.time()
    dataset_new.shingle_lookup.keys()
    for i in range(0,10000):
        dataset_new.__getitem__(i)
    print(time.time()-t0)

    t = dataset_new.pq_dataset.to_table().to_pandas()
    my_table.filter(pa.compute.equal(my_table['col1'], 'foo'))

    grid = grids.NGridBetter(config['grid_bounds'], kwargs['grid_s_size'])
    grid.add_grid_content(data_utils.map_from_deeptte(dataset.content,["locationtime","x","y","speed_m_s","bearing"]))
    grid.build_cell_lookup()
    dataset.grid = grid
    
    print(len(dataset.content))
    print(sum([len(x['lon']) for x in dataset.content]))

    # Prefetch
    # reverse sort
    # cache time values

    x_idx = np.concatenate([np.array(sample['x']) for sample in dataset.content])
    y_idx = np.concatenate([np.array(sample['y']) for sample in dataset.content])
    locationtime = np.concatenate([np.array(sample['locationtime']) for sample in dataset.content])
    n_points = 3
    x_idx, y_idx = grid.digitize_points(x_idx, y_idx)
    # grid.get_recent_points(x_idx, y_idx, locationtime, 3)

    t0 = time.time()
    num_cells = len(x_idx)
    cell_points = np.empty((num_cells, n_points, 6))
    cell_points.fill(np.nan)
    for i, (x, y, t) in enumerate(zip(x_idx, y_idx, locationtime)):
        # Get lookup values for every pt/cell, default empty array
        # If there are no points, or if buffer goes off edge of grid, return empty
        cell = grid.cell_lookup.get((x,y), np.array([]))
        if cell.size==0:
            continue
        else:
            # Get only n most recent values that occurred before the pt locationtime
            # These 3 are the expensive calls:
            idx = np.searchsorted(cell[:,0],t)
            points = cell[:idx,:][-n_points:][::-1]
        cell_points[i,:points.shape[0],:5] = points
    # Add obs_age feature
    cell_points[:,:,-1] = np.repeat(np.expand_dims(np.array(locationtime),1),n_points,1) - cell_points[:,:,0]
    train_time = (time.time() - t0)
    print(train_time)








    model = ff.FF(
        "FF",
        n_features=12,
        hidden_size=kwargs['HIDDEN_SIZE'],
        batch_size=kwargs['BATCH_SIZE'],
        embed_dict=embed_dict,
        collate_fn=data_loader.basic_collate,
        device=device
    ).to(device)

    t0 = time.time()
    dataset.add_grid_features = False
    loader = DataLoader(dataset, collate_fn=model.collate_fn, batch_size=kwargs['BATCH_SIZE'], pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)
    for batch in loader:
        len(batch)
    train_time = (time.time() - t0)
    print(train_time)

    model = ff.FF_GRID(
        "FF_NGRID_IND",
        n_features=12,
        n_grid_features=3*3*5*5,
        hidden_size=kwargs['HIDDEN_SIZE'],
        grid_compression_size=8,
        batch_size=kwargs['BATCH_SIZE'],
        embed_dict=embed_dict,
        collate_fn=data_loader.basic_grid_collate,
        device=device
    ).to(device)

    t0 = time.time()
    dataset.add_grid_features = True
    loader = DataLoader(dataset, collate_fn=model.collate_fn, batch_size=kwargs['BATCH_SIZE'], pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)
    for batch in loader:
        len(batch)
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