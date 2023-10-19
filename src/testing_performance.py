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
import lightning.pytorch as pl


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

    num_workers=0
    pin_memory=False
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
            'batch_size': 512,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout_rate': .2
        },
        'CONV': {
            'batch_size': 512,
            'hidden_size': 64,
            'num_layers': 3,
            'dropout_rate': .1
        },
        'GRU': {
            'batch_size': 512,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout_rate': .05
        },
        'TRSF': {
            'batch_size': 512,
            'hidden_size': 512,
            'num_layers': 6,
            'dropout_rate': .1
        },
        'DEEPTTE': {
            'batch_size': 512
        }
    }
    

    model_type="TRSF"
    skip_gtfs=False
    base_folder = f"{run_folder}{train_network_folder}"
    model_folder = f"{run_folder}{train_network_folder}models/{model_type}/"
    fold_num=0
    grid_s_size=500
    holdout_routes=None


    with open(f"{base_folder}deeptte_formatted/train_summary_config.json", "r") as f:
        config = json.load(f)

    print(f"Building grid on fold training data")
    train_dataset = data_loader.LoadSliceDataset(f"{base_folder}deeptte_formatted/train", config, holdout_routes=holdout_routes, skip_gtfs=skip_gtfs)
    train_ngrid = grids.NGridBetter(config['grid_bounds'][0], grid_s_size)
    train_ngrid.add_grid_content(train_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    train_ngrid.build_cell_lookup()
    train_dataset.grid = train_ngrid

    for fold_num in range(5):
        base_model_list, nn_model = model_utils.make_one_model("TRSF", hyperparameter_dict=hyperparameter_dict, embed_dict=embed_dict, config=config, skip_gtfs=skip_gtfs, load_weights=True, weight_folder=f"{model_folder}logs/{model_type}/version_{fold_num}/checkpoints/", fold_num=fold_num)
        print(sum(p.numel() for p in nn_model.parameters()))
        model_names = []
        model_names.append(nn_model.model_name)
        print(f"Model name: {model_names}")
        print(f"NN model total parameters: {sum(p.numel() for p in nn_model.parameters())}")

        # Keep track of all model performances
        model_fold_results = {}
        for x in model_names:
            model_fold_results[x] = {
                "Labels":[],
                "Preds":[]
            }

        loader = DataLoader(train_dataset, batch_size=nn_model.batch_size, collate_fn=nn_model.collate_fn, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
        trainer = pl.Trainer(accelerator="cpu", limit_predict_batches=1)
        preds_and_labels = trainer.predict(model=nn_model, dataloaders=loader)
        preds = np.concatenate([p['out_agg'] for p in preds_and_labels])
        labels = np.concatenate([l['y_agg'] for l in preds_and_labels])
        model_fold_results[nn_model.model_name]["Labels"].extend(list(labels))
        model_fold_results[nn_model.model_name]["Preds"].extend(list(preds))
        mape_score = metrics.mean_absolute_percentage_error(model_fold_results[nn_model.model_name]["Labels"], model_fold_results[nn_model.model_name]["Preds"])
        print(mape_score)

    print(1)



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