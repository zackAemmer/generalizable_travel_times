#!/usr/bin python3


import gc
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler
from sklearn import metrics
from sklearn.model_selection import KFold

from models import avg_speed, conv, ff, persistent, rnn, schedule, transformer
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
        config = json.load(f)
    train_network_dataset = data_loader.GenericDataset(f"{run_folder}{train_network_folder}deeptte_formatted/test", config, holdout_routes=kwargs['holdout_routes'])
    test_network_dataset = data_loader.GenericDataset(f"{run_folder}{test_network_folder}deeptte_formatted/test", config, holdout_routes=kwargs['holdout_routes'])
    tune_network_dataset = data_loader.GenericDataset(f"{run_folder}{test_network_folder}deeptte_formatted/train", config, subset=kwargs['n_tune_samples'], holdout_routes=kwargs['holdout_routes'])
    holdout_route_dataset = data_loader.GenericDataset(f"{run_folder}{train_network_folder}deeptte_formatted/test", config, subset=kwargs['n_tune_samples'], holdout_routes=kwargs['holdout_routes'], keep_only_holdout=True)
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

        # Declare baseline models
        base_model_list = []
        base_model_list.append(data_utils.load_pkl(f"{run_folder}{train_network_folder}models/AVG_{fold_num}.pkl"))
        base_model_list.append(data_utils.load_pkl(f"{run_folder}{train_network_folder}models/SCH_{fold_num}.pkl"))
        base_model_list.append(data_utils.load_pkl(f"{run_folder}{train_network_folder}models/PER_TIM_{fold_num}.pkl"))
        # Declare neural network models
        nn_model_list = model_utils.make_all_models(HIDDEN_SIZE, BATCH_SIZE, embed_dict, device)
        nn_optimizer_list = [torch.optim.Adam(model.parameters(), lr=LEARN_RATE) for model in nn_model_list]
        # Load all model weights
        for m in nn_model_list:
            m = m.load_state_dict(torch.load(f"{run_folder}{train_network_folder}models/{m.model_name}_{fold_num}.pt"))
        # Summarize models in run
        model_names = [x.model_name for x in base_model_list]
        model_names.extend([x.model_name for x in nn_model_list])
        print(f"Model names: {model_names}")
        print(f"NN model total parameters: {[sum(p.numel() for p in m.parameters()) for m in nn_model_list]}")
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
        # Construct dataloaders for all models
        base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(train_network_dataset, train_network_sampler, BATCH_SIZE, NUM_WORKERS)
        # Test all models
        for model, loader in zip(base_model_list, base_dataloaders):
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Train_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Train_Preds"].extend(list(preds))
        for model, loader in zip(nn_model_list, nn_dataloaders):
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Train_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Train_Preds"].extend(list(preds))

        print(f"EXPERIMENT: DIFFERENT NETWORK")
        print(f"Evaluating {run_folder}{train_network_folder} on {test_network_folder}")
        # Construct dataloaders for all models
        base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(test_network_dataset, test_network_sampler, BATCH_SIZE, NUM_WORKERS)
        # Test all models
        for model, loader in zip(base_model_list, base_dataloaders):
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Test_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Test_Preds"].extend(list(preds))
        for model, loader in zip(nn_model_list, nn_dataloaders):
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Test_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Test_Preds"].extend(list(preds))

        print(f"EXPERIMENT: HOLDOUT ROUTES")
        print(f"Evaluating {run_folder}{train_network_folder} on holdout routes from {train_network_folder}")
        # Construct dataloaders for all models
        base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(holdout_route_dataset, holdout_route_sampler, BATCH_SIZE, NUM_WORKERS)
        # Test all models
        for model, loader in zip(base_model_list, base_dataloaders):
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Holdout_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Holdout_Preds"].extend(list(preds))
        for model, loader in zip(nn_model_list, nn_dataloaders):
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Holdout_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Holdout_Preds"].extend(list(preds))

        # Fine-tune each model, then test on a set from a different network
        print(f"EXPERIMENT: FINE TUNING")
        # Re-declare nn models and all model list
        nn_model_list = model_utils.make_all_models(HIDDEN_SIZE, BATCH_SIZE, embed_dict, device)
        nn_optimizer_list = [torch.optim.Adam(model.parameters(), lr=LEARN_RATE) for model in nn_model_list]
        # Load all model weights
        for m in nn_model_list:
            m = m.load_state_dict(torch.load(f"{run_folder}{train_network_folder}models/{m.model_name}_{fold_num}.pt"))
        for epoch in range(kwargs['TUNE_EPOCHS']):
            print(f"FOLD: {fold_num}, FINE TUNING EPOCH: {epoch}")
            # Construct dataloaders for all models
            base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(tune_network_dataset, tune_network_sampler, BATCH_SIZE, NUM_WORKERS)
            # Train all models
            for model, loader in zip(base_model_list, base_dataloaders):
                print(f"Training: {model.model_name}")
                model.train(loader, config)
            for model, loader, optimizer in zip(nn_model_list, nn_dataloaders, nn_optimizer_list):
                print(f"Training: {model.model_name}")
                avg_batch_loss = model_utils.train(model, loader, optimizer, LEARN_RATE)
        # Test all models on train network
        base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(train_network_dataset, train_network_sampler, BATCH_SIZE, NUM_WORKERS)
        for model, loader in zip(base_model_list, base_dataloaders):
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Tune_Train_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Tune_Train_Preds"].extend(list(preds))
        for model, loader in zip(nn_model_list, nn_dataloaders):
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Tune_Train_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Tune_Train_Preds"].extend(list(preds))
        # Test all models on test network
        base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(test_network_dataset, test_network_sampler, BATCH_SIZE, NUM_WORKERS)
        for model, loader in zip(base_model_list, base_dataloaders):
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Tune_Test_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Tune_Test_Preds"].extend(list(preds))
        for model, loader in zip(nn_model_list, nn_dataloaders):
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Tune_Test_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Tune_Test_Preds"].extend(list(preds))
        # Save tuned models
        print(f"Fold {fold_num} fine tuning complete, saving model states and metrics...")
        for model in nn_model_list:
            torch.save(model.state_dict(), f"{run_folder}{train_network_folder}models/{model.model_name}_tuned_{fold_num}.pt")

        # Fine-tune each model, then test on a set from a different network
        print(f"EXPERIMENT: FEATURE EXTRACTION")
        # Re-declare nn models and all model list
        nn_model_list = model_utils.make_all_models(HIDDEN_SIZE, BATCH_SIZE, embed_dict, device)
        nn_optimizer_list = [torch.optim.Adam(model.parameters(), lr=LEARN_RATE) for model in nn_model_list]
        # Load all model weights
        for m in nn_model_list:
            m = m.load_state_dict(torch.load(f"{run_folder}{train_network_folder}models/{m.model_name}_{fold_num}.pt"))
        for epoch in range(kwargs['TUNE_EPOCHS']):
            print(f"FOLD: {fold_num}, FEATURE EXTRACTION EPOCH: {epoch}")
            # Construct dataloaders for all models
            base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(tune_network_dataset, tune_network_sampler, BATCH_SIZE, NUM_WORKERS)
            # Train all models
            for model, loader in zip(base_model_list, base_dataloaders):
                print(f"Training: {model.model_name}")
                model.train(loader, config)
            for model, loader, optimizer in zip(nn_model_list, nn_dataloaders, nn_optimizer_list):
                print(f"Training: {model.model_name}")
                avg_batch_loss = model_utils.train(model, loader, optimizer, LEARN_RATE)
        # Test all models on train network
        base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(train_network_dataset, train_network_sampler, BATCH_SIZE, NUM_WORKERS)
        for model, loader in zip(base_model_list, base_dataloaders):
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Extract_Train_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Extract_Train_Preds"].extend(list(preds))
        for model, loader in zip(nn_model_list, nn_dataloaders):
            model_utils.set_feature_extraction(model)
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Extract_Train_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Extract_Train_Preds"].extend(list(preds))
        # Test all models on test network
        base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(test_network_dataset, test_network_sampler, BATCH_SIZE, NUM_WORKERS)
        for model, loader in zip(base_model_list, base_dataloaders):
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Extract_Test_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Extract_Test_Preds"].extend(list(preds))
        for model, loader in zip(nn_model_list, nn_dataloaders):
            model_utils.set_feature_extraction(model)
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Extract_Test_Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Extract_Test_Preds"].extend(list(preds))
        # Save tuned models
        print(f"Fold {fold_num} fine tuning complete, saving model states and metrics...")
        for model in nn_model_list:
            torch.save(model.state_dict(), f"{run_folder}{train_network_folder}models/{model.model_name}_extracted_{fold_num}.pt")

        #     # Train all models on each training file; split samples in each file by fold
        #     for tune_file in list(tune_file_list):
        #         # Load data and config for this training fold/file
        #         tune_data, _, ngrid = data_utils.load_fold_data(tune_data_folder, tune_file, fold_num, kwargs['n_folds'])
        #         ngrid_content = ngrid.get_fill_content()
        #         print(f"TUNE FILE: {tune_file}, {len(tune_data)} tune samples")
        #         with open(f"{train_data_folder}train_config.json", "r") as f:
        #             config = json.load(f)
        #         # Construct dataloaders
        #         base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(tune_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, holdout_routes=kwargs['holdout_routes'], combine=False, data_subset=kwargs['n_tune_samples'])
        #         # Train nn models
        #         for model, loader in zip(nn_model_list, nn_dataloaders):
        #             avg_batch_loss = model_utils.train(model, loader, kwargs['LEARN_RATE'])
        # # Save tuned models
        # print(f"Fold {fold_num} fine tuning complete, saving model states and metrics...")
        # for model in nn_model_list:
        #     torch.save(model.state_dict(), f"{run_folder}{train_network_folder}models/{model.model_name}_tuned_{fold_num}.pt")
        # # Combine base models with newly tuned models for testing
        # all_model_list = []
        # all_model_list.extend(base_model_list)
        # all_model_list.extend(nn_model_list)
        # # Retest each model on the original and generalization networks
        # print(f"Evaluating {run_folder}{train_network_folder} on {train_data_folder}")
        # for valid_file in train_file_list:
        #     print(f"VALIDATE FILE: {valid_file}")
        #     valid_data, ngrid = data_utils.load_all_data(train_data_folder, valid_file)
        #     ngrid_content = ngrid.get_fill_content()
        #     with open(f"{train_data_folder}train_config.json", "r") as f:
        #         config = json.load(f)
        #     print(f"Successfully loaded {len(valid_data)} testing samples.")
        #     # Construct dataloaders for all models
        #     dataloaders = model_utils.make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, holdout_routes=kwargs['holdout_routes'], data_subset=kwargs['data_subset'])
        #     # Test all models
        #     for model, loader in zip(all_model_list, dataloaders):
        #         labels, preds = model.evaluate(loader, config)
        #         model_fold_results[model.model_name]["Tune_Train_Labels"].extend(list(labels))
        #         model_fold_results[model.model_name]["Tune_Train_Preds"].extend(list(preds))
        # print(f"Evaluating {run_folder}{train_network_folder} on {test_data_folder}")
        # for valid_file in test_file_list:
        #     print(f"VALIDATE FILE: {valid_file}")
        #     valid_data, ngrid = data_utils.load_all_data(test_data_folder, valid_file)
        #     ngrid_content = ngrid.get_fill_content()
        #     with open(f"{train_data_folder}train_config.json", "r") as f:
        #         config = json.load(f)
        #     print(f"Successfully loaded {len(valid_data)} testing samples.")
        #     # Construct dataloaders for all models
        #     dataloaders = model_utils.make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, holdout_routes=kwargs['holdout_routes'], data_subset=kwargs['data_subset'])
        #     # Test all models
        #     for model, loader in zip(all_model_list, dataloaders):
        #         labels, preds = model.evaluate(loader, config)
        #         model_fold_results[model.model_name]["Tune_Test_Labels"].extend(list(labels))
        #         model_fold_results[model.model_name]["Tune_Test_Preds"].extend(list(preds))

        # # Fine-tune each model, then test on a set from a different network
        # print(f"EXPERIMENT: FEATURE EXTRACTION")
        # # Re-declare nn models and all model list
        # nn_model_list = model_utils.make_all_models(HIDDEN_SIZE, BATCH_SIZE, embed_dict, device)
        # for m in nn_model_list:
        #     m = m.load_state_dict(torch.load(f"{run_folder}{train_network_folder}models/{m.model_name}_{fold_num}.pt"))
        # for epoch in range(kwargs['TUNE_EPOCHS']):
        #     print(f"FOLD: {fold_num}, FEATURE EXTRACTION EPOCH: {epoch}")
        #     # Train all models on each training file; split samples in each file by fold
        #     for tune_file in list(tune_file_list):
        #         # Load data and config for this training fold/file
        #         tune_data, _, ngrid = data_utils.load_fold_data(tune_data_folder, tune_file, fold_num, kwargs['n_folds'])
        #         ngrid_content = ngrid.get_fill_content()
        #         print(f"TUNE FILE: {tune_file}, {len(tune_data)} tune samples")
        #         with open(f"{train_data_folder}train_config.json", "r") as f:
        #             config = json.load(f)
        #         # Construct dataloaders
        #         base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(tune_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, holdout_routes=kwargs['holdout_routes'], combine=False, data_subset=kwargs['n_tune_samples'])
        #         # Train nn models
        #         for model, loader in zip(nn_model_list, nn_dataloaders):
        #             model_utils.set_feature_extraction(model)
        #             avg_batch_loss = model_utils.train(model, loader, kwargs['LEARN_RATE'])
        # # Save tuned models
        # print(f"Fold {fold_num} feature extraction complete, saving model states and metrics...")
        # for model in nn_model_list:
        #     torch.save(model.state_dict(), f"{run_folder}{train_network_folder}models/{model.model_name}_extracted_{fold_num}.pt")
        # # Combine base models with newly tuned models for testing
        # all_model_list = []
        # all_model_list.extend(base_model_list)
        # all_model_list.extend(nn_model_list)
        # # Retest each model on the original and generalization networks
        # print(f"Evaluating {run_folder}{train_network_folder} on {train_data_folder}")
        # for valid_file in train_file_list:
        #     print(f"VALIDATE FILE: {valid_file}")
        #     valid_data, ngrid = data_utils.load_all_data(train_data_folder, valid_file)
        #     ngrid_content = ngrid.get_fill_content()
        #     with open(f"{train_data_folder}train_config.json", "r") as f:
        #         config = json.load(f)
        #     print(f"Successfully loaded {len(valid_data)} testing samples.")
        #     # Construct dataloaders for all models
        #     dataloaders = model_utils.make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, holdout_routes=kwargs['holdout_routes'], data_subset=kwargs['data_subset'])
        #     # Test all models
        #     for model, loader in zip(all_model_list, dataloaders):
        #         labels, preds = model.evaluate(loader, config)
        #         model_fold_results[model.model_name]["Extract_Train_Labels"].extend(list(labels))
        #         model_fold_results[model.model_name]["Extract_Train_Preds"].extend(list(preds))
        # print(f"Evaluating {run_folder}{train_network_folder} on {test_data_folder}")
        # for valid_file in test_file_list:
        #     print(f"VALIDATE FILE: {valid_file}")
        #     valid_data, ngrid = data_utils.load_all_data(test_data_folder, valid_file)
        #     ngrid_content = ngrid.get_fill_content()
        #     with open(f"{train_data_folder}train_config.json", "r") as f:
        #         config = json.load(f)
        #     print(f"Successfully loaded {len(valid_data)} testing samples.")
        #     # Construct dataloaders for all models
        #     dataloaders = model_utils.make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, holdout_routes=kwargs['holdout_routes'], data_subset=kwargs['data_subset'])
        #     # Test all models
        #     for model, loader in zip(all_model_list, dataloaders):
        #         labels, preds = model.evaluate(loader, config)
        #         model_fold_results[model.model_name]["Extract_Test_Labels"].extend(list(labels))
        #         model_fold_results[model.model_name]["Extract_Test_Preds"].extend(list(preds))

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

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_experiments(
        run_folder="./results/debug/",
        train_network_folder="kcm/",
        test_network_folder="atb/",
        tune_network_folder="atb/",
        TUNE_EPOCHS=10,
        BATCH_SIZE=64,
        LEARN_RATE=1e-3,
        HIDDEN_SIZE=32,
        data_subset=.1,
        n_tune_samples=100,
        n_folds=3,
        holdout_routes=[100252,100139,102581,100341,102720]
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_experiments(
        run_folder="./results/debug/",
        train_network_folder="atb/",
        test_network_folder="kcm/",
        tune_network_folder="kcm/",
        TUNE_EPOCHS=10,
        BATCH_SIZE=64,
        LEARN_RATE=1e-3,
        HIDDEN_SIZE=32,
        data_subset=.1,
        n_tune_samples=100,
        n_folds=3,
        holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"]
    )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_experiments(
    #     run_folder="./results/cross_attn/",
    #     train_network_folder="kcm/",
    #     test_network_folder="atb/",
    #     tune_network_folder="atb/",
    #     TUNE_EPOCHS=10,
    #     BATCH_SIZE=64,
    #     LEARN_RATE=1e-3,
    #     HIDDEN_SIZE=32,
    #     data_subset=.1,
    #     n_tune_samples=100,
    #     n_folds=5,
    #     holdout_routes=[100252,100139,102581,100341,102720]
    # )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_experiments(
    #     run_folder="./results/cross_attn/",
    #     train_network_folder="atb/",
    #     test_network_folder="kcm/",
    #     tune_network_folder="kcm/",
    #     TUNE_EPOCHS=10,
    #     BATCH_SIZE=64,
    #     LEARN_RATE=1e-3,
    #     HIDDEN_SIZE=32,
    #     data_subset=.1,
    #     n_tune_samples=100,
    #     n_folds=5,
    #     holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"]
    # )