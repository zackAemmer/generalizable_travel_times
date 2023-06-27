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
    with profile(activities=[ProfilerActivity.CPU]) as prof:

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

        # Set hyperparameters
        BATCH_SIZE = kwargs['BATCH_SIZE']
        HIDDEN_SIZE = kwargs['HIDDEN_SIZE']
        LEARN_RATE = kwargs['LEARN_RATE']

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

        with record_function("create dataset"):
            # Data loading and fold setup
            print(f"DATA: '{run_folder}{train_network_folder}deeptte_formatted/'")
            with open(f"{run_folder}{train_network_folder}deeptte_formatted/train_config.json", "r") as f:
                config = json.load(f)
            dataset = data_loader.GenericDataset(f"{run_folder}{train_network_folder}deeptte_formatted/train", config, subset=.3, holdout_routes=kwargs['holdout_routes'])
        with record_function("create grid"):
            grid = grids.NGridBetter(config['grid_bounds'], kwargs['grid_s_size'])
            grid.add_grid_content(data_utils.map_from_deeptte(dataset.content,["locationtime","x","y","speed_m_s","bearing"]))
            grid.build_cell_lookup()

        # with record_function("iterate dataset without grid"):
        #     t0 = time.time()
        #     dataset.add_grid_features = False
        #     for batch in dataset:
        #         x = len(batch['lngs'])
        #     train_time = (time.time() - t0)
        #     print(train_time)

        with record_function("map from deeptte"):
            d = data_utils.map_from_deeptte(dataset.content, ["locationtime","x","y","speed_m_s","bearing"])
        with record_function("digitize"):
            xbin_idxs, ybin_idxs = grid.digitize_points(d[:,1], d[:,2])
        with record_function("setup feature queries"):
            x_idxs = xbin_idxs
            y_idxs = ybin_idxs
            locationtimes = d[:,0]
            n_points=3
            buffer=2
            seq_len = len(x_idxs)
            grid_size = (2 * buffer) + 1
            # For every point, want grid buffer in x and y
            x_range = np.arange(grid_size)
            y_range = np.arange(grid_size)
            x_buffer_range = x_idxs[:, np.newaxis] - buffer + x_range
            y_buffer_range = y_idxs[:, np.newaxis] - buffer + y_range
            # For every 1d set of X,Y grid ranges, want a 2d buffer
            buffer_range = [np.meshgrid(arr1,arr2) for arr1,arr2 in zip(x_buffer_range,y_buffer_range)]
            # Each element in each list is total enumeration of x,y cell indices for 1 point
            x_queries = np.concatenate([x[0].flatten() for x in buffer_range])
            y_queries = np.concatenate([y[1].flatten() for y in buffer_range])
            x_queries = np.clip(x_queries,0,len(grid.xbins))
            y_queries = np.clip(y_queries,0,len(grid.ybins))
            # Limit to bounds of the grid
            t_queries = np.array(locationtimes).repeat(grid_size*grid_size)
        with record_function("set up response to queries"):
            x_idx = x_queries
            y_idx = y_queries
            locationtime = t_queries
            num_cells = len(x_idx)
            cell_points = np.empty((num_cells, n_points, 6))
            cell_points.fill(np.nan)
        with record_function("query dict"):
            for i, (x,y,t) in enumerate(zip(x_idx, y_idx, locationtime)):
                # Get lookup values for every pt/cell
                cell = grid.cell_lookup[(x,y)]
                if cell.size==0:
                    continue
                else:
                    idx = np.searchsorted(cell[:,0],t)
                    points = cell[:idx,:][-n_points:][::-1]

                cell_points[i,:points.shape[0],:5] = points

        # with record_function("add obs age"):
        #     # Add obs_age feature
        #     cell_points[:,:,-1] = np.repeat(np.expand_dims(np.array(locationtime),1),n_points,1) - cell_points[:,:,0]
        #     n_recent_points = cell_points
        # with record_function("cleanup"):
        #     n_recent_points = n_recent_points.reshape((seq_len,grid_size,grid_size,n_points,6))
        #     # TxXxYxNxC -> TxCxYxNxX -> TxCxNxYxX
        #     n_recent_points = np.swapaxes(n_recent_points, 1, 4)
        #     grid_features = np.swapaxes(n_recent_points, 2, 3)
        # with record_function("apply grid norm"):
        #     grid_features = data_loader.apply_grid_normalization(grid_features, config)

            # dataset.grid = grid
            # dataset.add_grid_features = True
            # t0 = time.time()
            # for batch in dataset:
            #     x = len(batch['lngs'])
            # train_time = (time.time() - t0)
            # print(train_time)

    prof.export_chrome_trace("trace.json")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))


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