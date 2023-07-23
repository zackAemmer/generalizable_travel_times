#!/usr/bin python3


import json
import os
import random
import shutil
import time

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner
from sklearn import metrics
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from models import grids
from utils import data_loader, data_utils, model_utils


def run_models(run_folder, network_folder, **kwargs):
    """
    Train each of the specified models on bus data found in the data folder.
    The data folder is generated using prepare_run.py.
    The data in the folder will have many attributes, not all necessariliy used.
    Use k-fold cross validation to split the data n times into train/test.
    The test set is used as validation during the training process.
    Model accuracy is measured at the end of each folds.
    Save the resulting models, and the data generated during training to the results folder.
    These are then analyzed in a jupyter notebook.
    """

    print("="*30)
    print(f"RUN MODELS: '{run_folder}'")
    print(f"NETWORK: '{network_folder}'")

    NUM_WORKERS=0
    PIN_MEMORY=False

    # Create folder structure; delete older results
    base_folder = f"{run_folder}{network_folder}"
    if "model_results_temp.pkl" in os.listdir(f"{base_folder}"):
        os.remove(f"{base_folder}model_results_temp.pkl")
    if "model_results.pkl" in os.listdir(f"{base_folder}"):
        os.remove(f"{base_folder}model_results.pkl")
    if "model_generalization_results.pkl" in os.listdir(f"{base_folder}"):
        os.remove(f"{base_folder}model_generalization_results.pkl")
    if "param_search_dict.pkl" in os.listdir(f"{base_folder}"):
        os.remove(f"{base_folder}param_search_dict.pkl")
    if "param_search_dict_sample.pkl" in os.listdir(f"{base_folder}"):
        os.remove(f"{base_folder}param_search_dict_sample.pkl")
    try:
        shutil.rmtree(f"{base_folder}models")
    except:
        print("Model folder not found to remove")
    try:
        shutil.rmtree(f"{base_folder}logs")
    except:
        print("Log folder not found to remove")
    os.mkdir(f"{base_folder}models")
    os.mkdir(f"{base_folder}logs")

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
    # Sample parameter values for random search
    if kwargs['is_param_search']:
        hyperparameter_sample_dict = {
            'n_param_samples': 1,
            'batch_size': [512],
            'hidden_size': [16, 32, 64, 128],
            'num_layers': [2, 3, 4, 5],
            'dropout_rate': [.05, .1, .2, .4]
        }
        hyperparameter_dict = model_utils.random_param_search(hyperparameter_sample_dict, ["FF","CONV","GRU","TRSF"])
        data_utils.write_pkl(hyperparameter_sample_dict, f"{base_folder}param_search_dict.pkl")
        data_utils.write_pkl(hyperparameter_dict, f"{base_folder}param_search_dict_sample.pkl")
    # Manually specified run without testing hyperparameters
    else:
        hyperparameter_dict = {
            'FF': {
                'batch_size': 512,
                'hidden_size': 32,
                'num_layers': 3,
                'dropout_rate': .2
            },
            'CONV': {
                'batch_size': 512,
                'hidden_size': 64,
                'num_layers': 4,
                'dropout_rate': .4
            },
            'GRU': {
                'batch_size': 512,
                'hidden_size': 16,
                'num_layers': 2,
                'dropout_rate': .4
            },
            'TRSF': {
                'batch_size': 512,
                'hidden_size': 32,
                'num_layers': 4,
                'dropout_rate': .4
            },
            'DEEPTTE': {
                'batch_size': 10
            }
        }

    # Data loading and fold setup
    with open(f"{run_folder}{network_folder}deeptte_formatted/train_summary_config.json", "r") as f:
        config = json.load(f)
    train_dataset = data_loader.LoadSliceDataset(f"{run_folder}{network_folder}deeptte_formatted/train", config, holdout_routes=kwargs['holdout_routes'], skip_gtfs=kwargs['skip_gtfs'])
    test_dataset = data_loader.LoadSliceDataset(f"{run_folder}{network_folder}deeptte_formatted/test", config, holdout_routes=kwargs['holdout_routes'], skip_gtfs=kwargs['skip_gtfs'])
    splits = KFold(kwargs['n_folds'], shuffle=True, random_state=0)
    run_results = []

    # Run full training process for each model during each validation fold
    for fold_num, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
        print("="*30)
        print(f"BEGIN FOLD: {fold_num}")

        # Random samplers for indices from this fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # Build grid using all data from this fold; could overfit but seems not to
        print(f"Building grid on fold training data")
        train_ngrid = grids.NGridBetter(config['grid_bounds'][0], kwargs['grid_s_size'])
        train_ngrid.add_grid_content(train_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
        train_ngrid.build_cell_lookup()
        train_dataset.grid = train_ngrid
        print(f"Building grid on fold testing data")
        test_ngrid = grids.NGridBetter(config['grid_bounds'][0], kwargs['grid_s_size'])
        test_ngrid.add_grid_content(test_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
        test_ngrid.build_cell_lookup()
        test_dataset.grid = test_ngrid

        # Declare models
        if kwargs['is_param_search']:
            model_list = model_utils.make_param_search_models(hyperparameter_dict, embed_dict, config)
        elif not kwargs['skip_gtfs']:
            model_list = model_utils.make_all_models(hyperparameter_dict, embed_dict, config)
        else:
            model_list = model_utils.make_all_models_nosch(hyperparameter_dict, embed_dict, config)
        model_names = [m.model_name for m in model_list]
        print(f"Model names: {model_names}")
        print(f"Model total parameters: {[sum(p.numel() for p in m.parameters()) for m in model_list if m.is_nn]}")

        # Keep track of all model performances
        model_fold_results = {}
        for x in model_names:
            model_fold_results[x] = {
                "Labels":[],
                "Preds":[]
            }

        # Total run samples
        print(f"Training on {len(train_dataset)} samples, testing on {len(test_dataset)} samples")

        # Train all models
        for model in model_list:
            print(f"Network {network_folder} Fold {fold_num} Model {model.model_name}")
            train_dataset.add_grid_features = model.requires_grid
            test_dataset.add_grid_features = model.requires_grid
            train_loader = DataLoader(train_dataset, batch_size=model.batch_size, sampler=train_sampler, collate_fn=model.collate_fn, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            val_loader = DataLoader(train_dataset, batch_size=model.batch_size, sampler=val_sampler, collate_fn=model.collate_fn, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            test_loader = DataLoader(test_dataset, batch_size=model.batch_size, collate_fn=model.collate_fn, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            if not model.is_nn:
                # Train base model on all fold data
                model.train(train_loader, config)
                print(f"Fold final evaluation for: {model.model_name}")
                labels, preds = model.evaluate(test_loader, config)
                model_fold_results[model.model_name]["Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Preds"].extend(list(preds))
                data_utils.write_pkl(model, f"{base_folder}models/{model.model_name}_{fold_num}.pkl")
            else:
                trainer = pl.Trainer(
                    limit_val_batches=.50,
                    limit_test_batches=.50,
                    check_val_every_n_epoch=2,
                    max_epochs=50,
                    min_epochs=10,
                    accelerator="cpu",
                    logger=CSVLogger(save_dir=f"{run_folder}{network_folder}logs/", name=model.model_name),
                    callbacks=[EarlyStopping(monitor=f"{model.model_name}_valid_loss", min_delta=.01, patience=3)]
                    # profiler="simple"
                )
                trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                trainer.test(model=model, dataloaders=test_loader)
                preds_and_labels = trainer.predict(model=model, dataloaders=test_loader)
                preds = np.concatenate([p[0] for p in preds_and_labels])
                labels = np.concatenate([l[1] for l in preds_and_labels])
                model_fold_results[model.model_name]["Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Preds"].extend(list(preds))

        # Save fold results
        run_results.append(model_fold_results)

    # Save full run results
    data_utils.write_pkl(run_results, f"{base_folder}model_results.pkl")
    print(f"MODEL RUN COMPLETED '{base_folder}'")


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    # DEBUG
    run_models(
        run_folder="./results/debug/",
        network_folder="kcm/",
        grid_s_size=500,
        n_folds=2,
        holdout_routes=[100252,100139,102581,100341,102720],
        skip_gtfs=False,
        is_param_search=False
    )
    run_models(
        run_folder="./results/debug/",
        network_folder="atb/",
        grid_s_size=500,
        n_folds=2,
        holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"],
        skip_gtfs=False,
        is_param_search=False
    )
    # DEBUG MIXED
    run_models(
        run_folder="./results/debug_nosch/",
        network_folder="kcm_atb/",
        grid_s_size=500,
        n_folds=2,
        holdout_routes=None,
        skip_gtfs=True,
        is_param_search=False
    )

    # # PARAM SEARCH
    # run_models(
    #     run_folder="./results/param_search/",
    #     network_folder="kcm/",
    #     grid_s_size=500,
    #     n_folds=3,
    #     holdout_routes=None,
    #     skip_gtfs=False,
    #     is_param_search=True
    # )
    # run_models(
    #     run_folder="./results/param_search/",
    #     network_folder="atb/",
    #     grid_s_size=500,
    #     n_folds=3,
    #     holdout_routes=None,
    #     skip_gtfs=False,
    #     is_param_search=True
    # )

    # # FULL RUN
    # run_models(
    #     run_folder="./results/full_run/",
    #     network_folder="kcm/",
    #     grid_s_size=500,
    #     n_folds=5,
    #     holdout_routes=[100252,100139,102581,100341,102720],
    #     skip_gtfs=False,
    #     is_param_search=False
    # )
    # run_models(
    #     run_folder="./results/full_run/",
    #     network_folder="atb/",
    #     grid_s_size=500,
    #     n_folds=5,
    #     holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"],
    #     skip_gtfs=False,
    #     is_param_search=False
    # )
    # # FULL RUN MIXED
    # run_models(
    #     run_folder="./results/full_run_nosch/",
    #     network_folder="kcm_atb/",
    #     grid_s_size=500,
    #     n_folds=5,
    #     holdout_routes=None,
    #     skip_gtfs=True,
    #     is_param_search=False
    # )