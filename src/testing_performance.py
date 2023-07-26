#!/usr/bin python3


import gc
import itertools
import json
import os
import random
import time
import h5py
import cProfile
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn import metrics
from tabulate import tabulate
from utils import data_loader

from models import avg_speed, conv, ff, grids, persistent, rnn, schedule, transformer
from utils import data_utils, model_utils

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

import os.path as path

def run(run_folder, train_network_folder, **kwargs):
    # with profile(activities=[ProfilerActivity.CPU]) as prof:
    print("="*30)
    print(f"RUN MODELS: '{run_folder}'")
    print(f"NETWORK: '{train_network_folder}'")

    NUM_WORKERS=4
    PIN_MEMORY=False
    embed_dict = {
        'timeID': {
            'vocab_size': 1440,
            'embed_dims': 8
        },
        'weekID': {
            'vocab_size': 7,
            'embed_dims': 3
        }
    }
    hyperparameter_dict = {
        'FF': {
            'batch_size': 1024,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout_rate': .2
        },
        'CONV': {
            'batch_size': 1024,
            'hidden_size': 64,
            'num_layers': 3,
            'dropout_rate': .1
        },
        'GRU': {
            'batch_size': 1024,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout_rate': .05
        },
        'TRSF': {
            'batch_size': 1024,
            'hidden_size': 64,
            'num_layers': 3,
            'dropout_rate': .1
        },
        'DEEPTTE': {
            'batch_size': 1024
        }
    }
    
    with open(f"{run_folder}{train_network_folder}deeptte_formatted/train_summary_config.json", "r") as f:
        config = json.load(f)
    train_dataset = data_loader.LoadSliceDataset(f"{run_folder}{train_network_folder}deeptte_formatted/test", config)
    test_dataset = data_loader.LoadSliceHasGridDataset(f"{run_folder}{train_network_folder}deeptte_formatted/test", config, grid_bounds=config['grid_bounds'][0], grid_s_size=kwargs['grid_s_size'])
    train_dataset.add_grid_features = True
    test_dataset.add_grid_features = True

    train_ngrid = grids.NGridBetter(config['grid_bounds'][0], kwargs['grid_s_size'])
    train_ngrid.add_grid_content(train_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    train_ngrid.build_cell_lookup()
    train_dataset.grid = train_ngrid
    
    train_loader = DataLoader(train_dataset, batch_size=1024, collate_fn=data_loader.sequential_grid_collate, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, multiprocessing_context="fork")
    test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=data_loader.sequential_grid_collate, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, multiprocessing_context="fork")

    t0 = time.time()
    res = [x for x in train_loader]
    # for i in range(1024):
    #     res = train_dataset.__getitem__(i)
    print(time.time()-t0)

    # t0 = time.time()
    # res = [x for x in test_loader]
    # # for i in range(1024):
    # #     res = test_dataset.__getitem__(i)
    # print(time.time()-t0)

    # x_idxs = np.concatenate([train_dataset.__getitem__(i)['samp'][:,7] for i in range(10000)])
    # y_idxs = np.concatenate([train_dataset.__getitem__(i)['samp'][:,8] for i in range(10000)])
    # locationtimes = np.concatenate([train_dataset.__getitem__(i)['samp'][:,4] for i in range(10000)])

    # with cProfile.Profile() as pr:
    #     x_idxs, y_idxs = train_ngrid.digitize_points(x_idxs, y_idxs)
    #     buffer=1
    #     n_points=3
        
        # seq_len = len(x_idxs)
        # grid_size = (2 * buffer) + 1
        # # For every point, want grid buffer in x and y
        # x_idxs_low = x_idxs - 1
        # x_idxs_high = x_idxs + 1
        # y_idxs_low = y_idxs - 1
        # y_idxs_high = y_idxs + 1
        # x_idxs = np.column_stack([x_idxs_low, x_idxs, x_idxs_high])
        # y_idxs = np.column_stack([y_idxs_low, y_idxs, y_idxs_high])
        # x_range = np.arange(grid_size)
        # y_range = np.arange(grid_size)
        # x_buffer_range = x_idxs[:, np.newaxis] - buffer + x_range
        # y_buffer_range = y_idxs[:, np.newaxis] - buffer + y_range
        # # For every 1d set of X,Y grid ranges, want a 2d buffer
        # buffer_range = [np.meshgrid(arr1,arr2) for arr1,arr2 in zip(x_buffer_range,y_buffer_range)] # SLOW
        # # Each element in each list is total enumeration of x,y cell indices for 1 point
        # x_queries = np.concatenate([x[0].flatten() for x in buffer_range]) #KINDA SLOW
        # y_queries = np.concatenate([y[1].flatten() for y in buffer_range]) #KINDA SLOW
        # t_queries = np.array(locationtimes).repeat(grid_size*grid_size)
        
    #     x_queries = x_idxs
    #     y_queries = y_idxs
    #     t_queries = np.array(locationtimes)


    #     locationtime_dict = defaultdict(list)
    #     order_dict = defaultdict(list)
    #     # Create lookup for unique cells, to time values that will be searched
    #     for i, (x, y, t) in enumerate(zip(x_queries, y_queries, t_queries)):
    #         locationtime_dict[(x,y)].append(t)
    #         order_dict[(x,y)].append(i)
    #     # Get point values for every unique cell, at the required times
    #     res_dict = {}
    #     for k in list(locationtime_dict.keys()):
    #         # Want to get a set of n_points for every locationtime recorded in this cell
    #         cell_res = np.full((len(locationtime_dict[k]), n_points, 5), np.nan)
    #         # Get all points for this grid cell
    #         cell = train_ngrid.cell_lookup.get(k, np.array([]))
    #         if len(cell)!=0:
    #             # Get the index of each locationtime that we need for this grid cell
    #             t_idxs = np.searchsorted(cell[:,0], np.array(locationtime_dict[k]))
    #             # Record the index through index-n_points for each locationtime that we need for this grid cell
    #             for i,n_back in enumerate(range(n_points)):
    #                 idx_back = t_idxs - n_back
    #                 # Record which points should be filled with nan
    #                 mask = idx_back < 0
    #                 # Clip so that operation can still be performed
    #                 idx_back = np.clip(idx_back, a_min=0, a_max=len(cell)-1)
    #                 cell_res[:,i,:] = cell[idx_back]
    #                 # Fill nans (instead of repeating the first cell value), this is more informative
    #                 cell_res[mask] = np.nan
    #             # Save all cell results
    #         res_dict[k] = cell_res
    #     # Reconstruct final result in the correct order (original locationtimes have been split among dict keys)
    #     cell_points = np.full((len(t_queries), n_points, 6), np.nan)
    #     for k in order_dict.keys():
    #         loc_order = order_dict[k]
    #         results = res_dict[k]
    #         cell_points[loc_order,:,:5] = results
    #     cell_points[:,:,-1] = np.repeat(np.expand_dims(np.array(t_queries),1),n_points,1) - cell_points[:,:,0]
    #     n_recent_points = cell_points


    #     n_recent_points = n_recent_points.reshape((seq_len,grid_size,grid_size,n_points,6))
    #     # TxXxYxNxC -> TxCxYxNxX -> TxCxNxYxX
    #     n_recent_points = np.swapaxes(n_recent_points, 1, 4)
    #     n_recent_points = np.swapaxes(n_recent_points, 2, 3)
    # pr.print_stats()






    # test_data = train_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']).values[:1024]
    # t0 = time.time()
    # for i in range(0,1024):
    #     d = np.expand_dims(test_data[i,:],0).repeat(30, axis=0)
    #     xbin_idxs, ybin_idxs = train_ngrid.digitize_points(d[:,2], d[:,3])
    #     grid_features = train_ngrid.get_grid_features(xbin_idxs, ybin_idxs, d[:,1])
    # print(time.time()-t0)
    
    # test_data = train_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing'])[:1024]
    # xbin_idxs, ybin_idxs = train_ngrid.digitize_points(test_data["x"], test_data["y"])
    # with cProfile.Profile() as pr:
    #     t0 = time.time()
    #     res = train_ngrid.get_grid_features(xbin_idxs, ybin_idxs, list(test_data["locationtime"]), 3)
    #     print(time.time()-t0)
    # pr.print_stats()
    # print("end")
    #.389 vs .696
    #.132 vs .467

    # if kwargs['is_param_search']:
    #     model_list = model_utils.make_param_search_models(hyperparameter_dict, embed_dict, config)
    # elif not kwargs['skip_gtfs']:
    #     model_list = model_utils.make_all_models(hyperparameter_dict, embed_dict, config)
    # else:
    #     model_list = model_utils.make_all_models_nosch(hyperparameter_dict, embed_dict, config)
    # model_names = [m.model_name for m in model_list]


    # train_loader = DataLoader(train_dataset, batch_size=model.batch_size, sampler=train_sampler, collate_fn=model.collate_fn, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    # val_loader = DataLoader(train_dataset, batch_size=model.batch_size, sampler=val_sampler, collate_fn=model.collate_fn, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    # test_loader = DataLoader(test_dataset, batch_size=model.batch_size, collate_fn=model.collate_fn, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            
    # trainer = pl.Trainer(
    #     limit_train_batches=.1,
    #     limit_val_batches=.10,
    #     limit_test_batches=.10,
    #     check_val_every_n_epoch=1,
    #     max_epochs=1,
    #     min_epochs=1,
    #     accelerator="cpu",
    #     logger=CSVLogger(save_dir=f"{run_folder}{network_folder}logs/", name=model.model_name),
    #     callbacks=[EarlyStopping(monitor=f"{model.model_name}_valid_loss", min_delta=.001, patience=3)],
    #     profiler=profiler
    # )
    # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
if __name__=="__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run(
        run_folder="./results/full_run/",
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