#!/usr/bin python3


import gc
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn import metrics

from models import grids
from utils import data_utils, data_loader, model_utils


def run_experiments(run_folder, train_network_folder, test_network_folder, tune_network_folder, **kwargs):
    print("="*30)
    print(f"RUN EXPERIMENTS: '{run_folder}'")
    print(f"TRAINED ON NETWORK: '{train_network_folder}'")
    print(f"TEST ON NETWORK: '{test_network_folder}'")

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

    # Data loading and fold setup
    print(f"DATA: '{run_folder}{train_network_folder}deeptte_formatted/'")
    with open(f"{run_folder}{train_network_folder}deeptte_formatted/test_config.json", "r") as f:
        train_network_config = json.load(f)
    with open(f"{run_folder}{test_network_folder}deeptte_formatted/test_config.json", "r") as f:
        test_network_config = json.load(f)
    with open(f"{run_folder}{test_network_folder}deeptte_formatted/train_config.json", "r") as f:
        tune_network_config = json.load(f)

    print(f"Building grid on validation data from training network")
    train_network_dataset = data_loader.GenericDataset(f"{run_folder}{train_network_folder}deeptte_formatted/test", train_network_config, holdout_routes=kwargs['holdout_routes'])
    train_network_ngrid = grids.NGridBetter(train_network_config['grid_bounds'],kwargs['grid_s_size'])
    train_network_ngrid.add_grid_content(data_utils.map_from_deeptte([x for i,x in enumerate(train_network_dataset.content) if True],["locationtime","x","y","speed_m_s","bearing"]))
    train_network_ngrid.build_cell_lookup()
    train_network_dataset.grid = train_network_ngrid
    print(f"Building grid on validation data from testing network")
    test_network_dataset = data_loader.GenericDataset(f"{run_folder}{test_network_folder}deeptte_formatted/test", test_network_config, holdout_routes=kwargs['holdout_routes'])
    test_network_ngrid = grids.NGridBetter(test_network_config['grid_bounds'],kwargs['grid_s_size'])
    test_network_ngrid.add_grid_content(data_utils.map_from_deeptte([x for i,x in enumerate(test_network_dataset.content) if True],["locationtime","x","y","speed_m_s","bearing"]))
    test_network_ngrid.build_cell_lookup()
    test_network_dataset.grid = test_network_ngrid
    print(f"Building tune grid on training data from testing network")
    tune_network_dataset = data_loader.GenericDataset(f"{run_folder}{test_network_folder}deeptte_formatted/train", tune_network_config, subset=kwargs['n_tune_samples'], holdout_routes=kwargs['holdout_routes'])
    tune_network_ngrid = grids.NGridBetter(tune_network_config['grid_bounds'],kwargs['grid_s_size'])
    tune_network_ngrid.add_grid_content(data_utils.map_from_deeptte([x for i,x in enumerate(tune_network_dataset.content) if True],["locationtime","x","y","speed_m_s","bearing"]))
    tune_network_ngrid.build_cell_lookup()
    tune_network_dataset.grid = tune_network_ngrid
    print(f"Building route holdout grid on validation data from training network")
    holdout_route_dataset = data_loader.GenericDataset(f"{run_folder}{train_network_folder}deeptte_formatted/test", train_network_config, subset=kwargs['n_tune_samples'], holdout_routes=kwargs['holdout_routes'], keep_only_holdout=True)
    holdout_network_ngrid = grids.NGridBetter(train_network_config['grid_bounds'],kwargs['grid_s_size'])
    holdout_network_ngrid.add_grid_content(data_utils.map_from_deeptte([x for i,x in enumerate(train_network_dataset.content) if True],["locationtime","x","y","speed_m_s","bearing"]))
    holdout_network_ngrid.build_cell_lookup()
    holdout_route_dataset.grid = holdout_network_ngrid

    run_results = []

    # Run experiments on each fold
    for fold_num in range(kwargs['n_folds']):
        print("="*30)
        print(f"BEGIN FOLD: {fold_num}")

        # Random samplers for indices from this fold
        train_network_sampler = SubsetRandomSampler(np.arange(len(train_network_dataset)))
        test_network_sampler = SubsetRandomSampler(np.arange(len(test_network_dataset)))
        tune_network_sampler = SubsetRandomSampler(np.arange(len(tune_network_dataset)))
        holdout_route_sampler = SubsetRandomSampler(np.arange(len(holdout_route_dataset)))

        # Declare models
        model_list = model_utils.make_all_models(HIDDEN_SIZE, BATCH_SIZE, embed_dict, device, train_network_config, load_weights=True, weight_folder=f"{run_folder}{train_network_folder}models/", fold_num=fold_num)
        model_names = [m.model_name for m in model_list]
        print(f"Model names: {model_names}")
        print(f"Model total parameters: {[sum(p.numel() for p in m.parameters()) for m in model_list if m.is_nn]}")
        # Keep track of all model performances
        model_fold_results = {}
        for x in model_names:
            model_fold_results[x] = {
                "Train_Labels":[],
                "Train_Preds":[],
                "Test_Labels":[],
                "Test_Preds":[],
                "Holdout_Labels":[],
                "Holdout_Preds":[],
                "Tune_Train_Labels":[],
                "Tune_Train_Preds":[],
                "Tune_Test_Labels":[],
                "Tune_Test_Preds":[],
                "Extract_Train_Labels":[],
                "Extract_Train_Preds":[],
                "Extract_Test_Labels":[],
                "Extract_Test_Preds":[]
            }

        print(f"EXPERIMENT: SAME NETWORK")
        print(f"Evaluating {run_folder}{train_network_folder} on {train_network_folder}")
        for model in model_list:
            print(f"Evaluating: {model.model_name}")
            train_network_dataset.add_grid_features = model.requires_grid
            loader = DataLoader(train_network_dataset, sampler=train_network_sampler, collate_fn=model.collate_fn, batch_size=BATCH_SIZE, pin_memory=[True if NUM_WORKERS>0 else False], num_workers=NUM_WORKERS)
            labels, preds = model.evaluate(loader, train_network_config)
            model_fold_results[model.model_name]["Train_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Train_Preds"].extend(list(preds))

        print(f"EXPERIMENT: DIFFERENT NETWORK")
        print(f"Evaluating {run_folder}{train_network_folder} on {test_network_folder}")
        for model in model_list:
            print(f"Evaluating: {model.model_name}")
            test_network_dataset.add_grid_features = model.requires_grid
            loader = DataLoader(test_network_dataset, sampler=test_network_sampler, collate_fn=model.collate_fn, batch_size=BATCH_SIZE, pin_memory=[True if NUM_WORKERS>0 else False], num_workers=NUM_WORKERS)
            labels, preds = model.evaluate(loader, test_network_config)
            model_fold_results[model.model_name]["Test_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Test_Preds"].extend(list(preds))

        print(f"EXPERIMENT: HOLDOUT ROUTES")
        print(f"Evaluating {run_folder}{train_network_folder} on holdout routes from {train_network_folder}")
        for model in model_list:
            print(f"Evaluating: {model.model_name}")
            holdout_route_dataset.add_grid_features = model.requires_grid
            loader = DataLoader(holdout_route_dataset, sampler=holdout_route_sampler, collate_fn=model.collate_fn, batch_size=BATCH_SIZE, pin_memory=[True if NUM_WORKERS>0 else False], num_workers=NUM_WORKERS)
            labels, preds = model.evaluate(loader, train_network_config)
            model_fold_results[model.model_name]["Holdout_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Holdout_Preds"].extend(list(preds))

        print(f"EXPERIMENT: FINE TUNING")
        # Re-declare models with original weights
        model_list = model_utils.make_all_models(HIDDEN_SIZE, BATCH_SIZE, embed_dict, device, train_network_config, load_weights=True, weight_folder=f"{run_folder}{train_network_folder}models/", fold_num=fold_num)
        # Tune model on tuning network
        for model in model_list:
            print(f"Tuning model {model.model_name} on {tune_network_folder}")
            tune_network_dataset.add_grid_features = model.requires_grid
            loader = DataLoader(tune_network_dataset, sampler=tune_network_sampler, collate_fn=model.collate_fn, batch_size=BATCH_SIZE, pin_memory=[True if NUM_WORKERS>0 else False], num_workers=NUM_WORKERS)
            if not model.is_nn:
                model.train(loader, train_network_config)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
                for epoch in range(kwargs['TUNE_EPOCHS']):
                    avg_batch_loss = model_utils.train(model, loader, optimizer)
        # Evaluate models on train/test networks
        for model in model_list:
            train_network_dataset.add_grid_features = model.requires_grid
            loader = DataLoader(train_network_dataset, sampler=train_network_sampler, collate_fn=model.collate_fn, batch_size=BATCH_SIZE, pin_memory=[True if NUM_WORKERS>0 else False], num_workers=NUM_WORKERS)
            labels, preds = model.evaluate(loader, train_network_config)
            model_fold_results[model.model_name]["Tune_Train_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Tune_Train_Preds"].extend(list(preds))
        for model in model_list:
            test_network_dataset.add_grid_features = model.requires_grid
            loader = DataLoader(test_network_dataset, sampler=test_network_sampler, collate_fn=model.collate_fn, batch_size=BATCH_SIZE, pin_memory=[True if NUM_WORKERS>0 else False], num_workers=NUM_WORKERS)
            labels, preds = model.evaluate(loader, test_network_config)
            model_fold_results[model.model_name]["Tune_Test_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Tune_Test_Preds"].extend(list(preds))
        # Save tuned models
        print(f"Fold {fold_num} fine tuning complete, saving model states and metrics...")
        for model in model_list:
            if model.is_nn:
                torch.save(model.state_dict(), f"{run_folder}{train_network_folder}models/{model.model_name}_tuned_{fold_num}.pt")
            else:
                model.save_to(f"{run_folder}{train_network_folder}models/{model.model_name}_tuned_{fold_num}.pkl")

        print(f"EXPERIMENT: FEATURE EXTRACTION")
        # Re-declare models with original weights
        model_list = model_utils.make_all_models(HIDDEN_SIZE, BATCH_SIZE, embed_dict, device, train_network_config, load_weights=True, weight_folder=f"{run_folder}{train_network_folder}models/", fold_num=fold_num)
        # Tune model on tuning network
        for model in model_list:
            print(f"Tuning model {model.model_name} on {tune_network_folder}")
            tune_network_dataset.add_grid_features = model.requires_grid
            loader = DataLoader(tune_network_dataset, sampler=tune_network_sampler, collate_fn=model.collate_fn, batch_size=BATCH_SIZE, pin_memory=[True if NUM_WORKERS>0 else False], num_workers=NUM_WORKERS)
            if not model.is_nn:
                model.train(loader, train_network_config)
            else:
                # Set only the final layer to train
                model_utils.set_feature_extraction(model)
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
                for epoch in range(kwargs['TUNE_EPOCHS']):
                    avg_batch_loss = model_utils.train(model, loader, optimizer)
        # Evaluate models on train/test networks
        for model in model_list:
            train_network_dataset.add_grid_features = model.requires_grid
            loader = DataLoader(train_network_dataset, sampler=train_network_sampler, collate_fn=model.collate_fn, batch_size=BATCH_SIZE, pin_memory=[True if NUM_WORKERS>0 else False], num_workers=NUM_WORKERS)
            labels, preds = model.evaluate(loader, train_network_config)
            model_fold_results[model.model_name]["Extract_Train_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Extract_Train_Preds"].extend(list(preds))
        for model in model_list:
            test_network_dataset.add_grid_features = model.requires_grid
            loader = DataLoader(test_network_dataset, sampler=test_network_sampler, collate_fn=model.collate_fn, batch_size=BATCH_SIZE, pin_memory=[True if NUM_WORKERS>0 else False], num_workers=NUM_WORKERS)
            labels, preds = model.evaluate(loader, test_network_config)
            model_fold_results[model.model_name]["Extract_Test_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Extract_Test_Preds"].extend(list(preds))
        # Save tuned models
        print(f"Fold {fold_num} fine tuning complete, saving model states and metrics...")
        for model in model_list:
            if model.is_nn:
                torch.save(model.state_dict(), f"{run_folder}{train_network_folder}models/{model.model_name}_tuned_{fold_num}.pt")
            else:
                model.save_to(f"{run_folder}{train_network_folder}models/{model.model_name}_tuned_{fold_num}.pkl")

        # Calculate various losses:
        fold_results = {
            "Model_Names": model_names,
            "Fold": fold_num,
            "Train_Losses": [],
            "Test_Losses": [],
            "Holdout_Losses": [],
            "Tune_Train_Losses": [],
            "Tune_Test_Losses": [],
            "Extract_Train_Losses": [],
            "Extract_Test_Losses": []
        }
        for mname in fold_results["Model_Names"]:
            _ = [mname]
            _.append(np.round(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Train_Labels"], model_fold_results[mname]["Train_Preds"]), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Train_Labels"], model_fold_results[mname]["Train_Preds"])), 2))
            _.append(np.round(metrics.mean_absolute_error(model_fold_results[mname]["Train_Labels"], model_fold_results[mname]["Train_Preds"]), 2))
            fold_results['Train_Losses'].append(_)
            _ = [mname]
            _.append(np.round(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Test_Labels"], model_fold_results[mname]["Test_Preds"]), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Test_Labels"], model_fold_results[mname]["Test_Preds"])), 2))
            _.append(np.round(metrics.mean_absolute_error(model_fold_results[mname]["Test_Labels"], model_fold_results[mname]["Test_Preds"]), 2))
            fold_results['Test_Losses'].append(_)
            _ = [mname]
            _.append(np.round(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Holdout_Labels"], model_fold_results[mname]["Holdout_Preds"]), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Holdout_Labels"], model_fold_results[mname]["Holdout_Preds"])), 2))
            _.append(np.round(metrics.mean_absolute_error(model_fold_results[mname]["Holdout_Labels"], model_fold_results[mname]["Holdout_Preds"]), 2))
            fold_results['Holdout_Losses'].append(_)
            _ = [mname]
            _.append(np.round(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Tune_Train_Labels"], model_fold_results[mname]["Tune_Train_Preds"]), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Tune_Train_Labels"], model_fold_results[mname]["Tune_Train_Preds"])), 2))
            _.append(np.round(metrics.mean_absolute_error(model_fold_results[mname]["Tune_Train_Labels"], model_fold_results[mname]["Tune_Train_Preds"]), 2))
            fold_results['Tune_Train_Losses'].append(_)
            _ = [mname]
            _.append(np.round(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Tune_Test_Labels"], model_fold_results[mname]["Tune_Test_Preds"]), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Tune_Test_Labels"], model_fold_results[mname]["Tune_Test_Preds"])), 2))
            _.append(np.round(metrics.mean_absolute_error(model_fold_results[mname]["Tune_Test_Labels"], model_fold_results[mname]["Tune_Test_Preds"]), 2))
            fold_results['Tune_Test_Losses'].append(_)
            _ = [mname]
            _.append(np.round(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Extract_Train_Labels"], model_fold_results[mname]["Extract_Train_Preds"]), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Extract_Train_Labels"], model_fold_results[mname]["Extract_Train_Preds"])), 2))
            _.append(np.round(metrics.mean_absolute_error(model_fold_results[mname]["Extract_Train_Labels"], model_fold_results[mname]["Extract_Train_Preds"]), 2))
            fold_results['Extract_Train_Losses'].append(_)
            _ = [mname]
            _.append(np.round(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Extract_Test_Labels"], model_fold_results[mname]["Extract_Test_Preds"]), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Extract_Test_Labels"], model_fold_results[mname]["Extract_Test_Preds"])), 2))
            _.append(np.round(metrics.mean_absolute_error(model_fold_results[mname]["Extract_Test_Labels"], model_fold_results[mname]["Extract_Test_Preds"]), 2))
            fold_results['Extract_Test_Losses'].append(_)

        # Save fold
        run_results.append(fold_results)

        # Clean memory at end of each fold
        gc.collect()
        if device==torch.device("cuda"):
            torch.cuda.empty_cache()

    # Save run results
    data_utils.write_pkl(run_results, f"{run_folder}{train_network_folder}model_generalization_results.pkl")
    print(f"EXPERIMENTS COMPLETED '{run_folder}{train_network_folder}'")


if __name__=="__main__":
    torch.set_default_dtype(torch.float)

    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_experiments(
    #     run_folder="./results/debug/",
    #     train_network_folder="kcm/",
    #     test_network_folder="atb/",
    #     tune_network_folder="atb/",
    #     TUNE_EPOCHS=4,
    #     BATCH_SIZE=64,
    #     LEARN_RATE=1e-3,
    #     HIDDEN_SIZE=32,
    #     grid_s_size=500,
    #     data_subset=.1,
    #     n_tune_samples=100,
    #     n_folds=2,
    #     holdout_routes=[100252,100139,102581,100341,102720]
    # )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_experiments(
        run_folder="./results/debug/",
        train_network_folder="atb/",
        test_network_folder="kcm/",
        tune_network_folder="kcm/",
        TUNE_EPOCHS=4,
        BATCH_SIZE=64,
        LEARN_RATE=1e-3,
        HIDDEN_SIZE=32,
        grid_s_size=500,
        data_subset=.1,
        n_tune_samples=100,
        n_folds=2,
        holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"]
    )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_experiments(
    #     run_folder="./results/small/",
    #     train_network_folder="kcm/",
    #     test_network_folder="atb/",
    #     tune_network_folder="atb/",
    #     TUNE_EPOCHS=10,
    #     BATCH_SIZE=64,
    #     LEARN_RATE=1e-3,
    #     HIDDEN_SIZE=32,
    #     data_subset=.1,
    #     n_tune_samples=100,
    #     n_folds=3,
    #     holdout_routes=[100252,100139,102581,100341,102720]
    # )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_experiments(
    #     run_folder="./results/small/",
    #     train_network_folder="atb/",
    #     test_network_folder="kcm/",
    #     tune_network_folder="kcm/",
    #     TUNE_EPOCHS=10,
    #     BATCH_SIZE=64,
    #     LEARN_RATE=1e-3,
    #     HIDDEN_SIZE=32,
    #     data_subset=.1,
    #     n_tune_samples=100,
    #     n_folds=3,
    #     holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"]
    # )