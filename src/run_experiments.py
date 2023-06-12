#!/usr/bin python3


import gc
import json
import os
import random

import numpy as np
import torch
from sklearn import metrics

from models import avg_speed, conv, ff, persistent, rnn, schedule, transformer
from utils import data_utils, model_utils


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

    # Get list of available test files
    train_data_folder = f"{run_folder}{train_network_folder}deeptte_formatted/"
    test_data_folder = f"{run_folder}{test_network_folder}deeptte_formatted/"
    tune_data_folder = f"{run_folder}{tune_network_folder}deeptte_formatted/"
    train_file_list = list(filter(lambda x: x[:4]=="test" and len(x)==5, os.listdir(train_data_folder)))
    train_file_list.sort()
    test_file_list = list(filter(lambda x: x[:4]=="test" and len(x)==5, os.listdir(test_data_folder)))
    test_file_list.sort()
    tune_file_list = list(filter(lambda x: x[:5]=="train" and len(x)==6, os.listdir(tune_data_folder)))
    tune_file_list.sort()
    print(f"TRAIN FILES: {train_data_folder} {train_file_list}")
    print(f"TEST FILES: {test_data_folder} {test_file_list}")
    print(f"TUNE FILES: {tune_data_folder} {tune_file_list}")
    print("="*30)

    run_results = []
    for fold_num in range(kwargs['n_folds']):
        print("="*30)
        print(f"BEGIN FOLD: {fold_num}")

        # Declare baseline models
        base_model_list = []
        base_model_list.append(data_utils.load_pkl(f"{run_folder}{train_network_folder}models/AVG_{fold_num}.pkl"))
        base_model_list.append(data_utils.load_pkl(f"{run_folder}{train_network_folder}models/SCH_{fold_num}.pkl"))
        base_model_list.append(data_utils.load_pkl(f"{run_folder}{train_network_folder}models/PER_TIM_{fold_num}.pkl"))

        # Declare neural network models
        nn_model_list = []
        nn_model_list.append(ff.FF(
            "FF",
            n_features=12,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(ff.FF_GRID(
            "FF_NGRID_IND",
            n_features=12,
            n_grid_features=3*3*5*5,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(rnn.GRU(
            "GRU",
            n_features=9,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(rnn.GRU_GRID(
            "GRU_NGRID_IND",
            n_features=9,
            n_grid_features=3*3*5*5,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(transformer.TRSF(
            "TRSF",
            n_features=9,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(transformer.TRSF_GRID(
            "TRSF_NGRID_IND",
            n_features=9,
            n_grid_features=3*3*5*5,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))

        all_model_list = []
        all_model_list.extend(base_model_list)
        all_model_list.extend(nn_model_list)

        print(f"Model names: {[m.model_name for m in nn_model_list]}")
        print(f"Model total parameters: {[sum(p.numel() for p in m.parameters()) for m in nn_model_list]}")

        # Load all model weights
        for m in nn_model_list:
            m = m.load_state_dict(torch.load(f"{run_folder}{train_network_folder}models/{m.model_name}_{fold_num}.pt"))

        # Test models on different networks
        model_fold_results = {}
        for x in all_model_list:
            model_fold_results[x.model_name] = {
                "Train_Labels":[],
                "Train_Preds":[],
                "Test_Labels":[],
                "Test_Preds":[],
                "Tune_Train_Labels":[],
                "Tune_Train_Preds":[],
                "Tune_Test_Labels":[],
                "Tune_Test_Preds":[],
                "Extract_Train_Labels":[],
                "Extract_Train_Preds":[],
                "Extract_Test_Labels":[],
                "Extract_Test_Preds":[]
            }

        # Test each model on a holdout validation set from the original training network
        print(f"EXPERIMENT: SAME NETWORK")
        print(f"Evaluating {run_folder}{train_network_folder} on {train_data_folder}")
        for valid_file in train_file_list:
            print(f"VALIDATE FILE: {valid_file}")
            valid_data, ngrid = data_utils.load_all_data(train_data_folder, valid_file)
            ngrid_content = ngrid.get_fill_content()
            with open(f"{train_data_folder}train_config.json", "r") as f:
                config = json.load(f)
            print(f"Successfully loaded {len(valid_data)} testing samples.")
            # Construct dataloaders for all models
            dataloaders = model_utils.make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, data_subset=kwargs['data_subset'])
            # Test all models
            for model, loader in zip(all_model_list, dataloaders):
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Train_Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Train_Preds"].extend(list(preds))

        # Test each model on a set from a different network
        print(f"EXPERIMENT: DIFFERENT NETWORK")
        print(f"Evaluating {run_folder}{train_network_folder} on {test_data_folder}")
        for valid_file in test_file_list:
            print(f"VALIDATE FILE: {valid_file}")
            valid_data, ngrid = data_utils.load_all_data(test_data_folder, valid_file)
            ngrid_content = ngrid.get_fill_content()
            with open(f"{train_data_folder}train_config.json", "r") as f:
                config = json.load(f)
            print(f"Successfully loaded {len(valid_data)} testing samples.")
            # Construct dataloaders for all models
            dataloaders = model_utils.make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, data_subset=kwargs['data_subset'])
            # Test all models
            for model, loader in zip(all_model_list, dataloaders):
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Test_Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Test_Preds"].extend(list(preds))

        # Fine-tune each model, then test on a set from a different network
        print(f"EXPERIMENT: FINE TUNING")
        for epoch in range(kwargs['TUNE_EPOCHS']):
            print(f"FOLD: {fold_num}, FINE TUNING EPOCH: {epoch}")
            # Train all models on each training file; split samples in each file by fold
            for tune_file in list(tune_file_list):
                # Load data and config for this training fold/file
                tune_data, _, ngrid = data_utils.load_fold_data(tune_data_folder, tune_file, fold_num, kwargs['n_folds'])
                ngrid_content = ngrid.get_fill_content()
                print(f"TUNE FILE: {tune_file}, {len(tune_data)} tune samples")
                with open(f"{train_data_folder}train_config.json", "r") as f:
                    config = json.load(f)
                # Construct dataloaders
                base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(tune_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, combine=False, data_subset=kwargs['n_tune_samples'])
                # Train all models
                for model, loader in zip(base_model_list, base_dataloaders):
                    model.train(loader, config)
                for model, loader in zip(nn_model_list, nn_dataloaders):
                    avg_batch_loss = model_utils.train(model, loader, kwargs['LEARN_RATE'])
        # Save tuned models
        print(f"Fold {fold_num} fine tuning complete, saving model states and metrics...")
        for model in base_model_list:
            model.save_to(f"{run_folder}{train_network_folder}models/{model.model_name}_tuned_{fold_num}.pkl")
        for model in nn_model_list:
            torch.save(model.state_dict(), f"{run_folder}{train_network_folder}models/{model.model_name}_tuned_{fold_num}.pt")
        # Retest each model on the original and generalization networks
        print(f"Evaluating {run_folder}{train_network_folder} on {train_data_folder}")
        for valid_file in train_file_list:
            print(f"VALIDATE FILE: {valid_file}")
            valid_data, ngrid = data_utils.load_all_data(train_data_folder, valid_file)
            ngrid_content = ngrid.get_fill_content()
            with open(f"{train_data_folder}train_config.json", "r") as f:
                config = json.load(f)
            print(f"Successfully loaded {len(valid_data)} testing samples.")
            # Construct dataloaders for all models
            dataloaders = model_utils.make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, data_subset=kwargs['data_subset'])
            # Test all models
            for model, loader in zip(all_model_list, dataloaders):
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Tune_Train_Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Tune_Train_Preds"].extend(list(preds))
        print(f"Evaluating {run_folder}{train_network_folder} on {test_data_folder}")
        for valid_file in test_file_list:
            print(f"VALIDATE FILE: {valid_file}")
            valid_data, ngrid = data_utils.load_all_data(test_data_folder, valid_file)
            ngrid_content = ngrid.get_fill_content()
            with open(f"{train_data_folder}train_config.json", "r") as f:
                config = json.load(f)
            print(f"Successfully loaded {len(valid_data)} testing samples.")
            # Construct dataloaders for all models
            dataloaders = model_utils.make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, data_subset=kwargs['data_subset'])
            # Test all models
            for model, loader in zip(all_model_list, dataloaders):
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Tune_Test_Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Tune_Test_Preds"].extend(list(preds))

        # Fine-tune each model, then test on a set from a different network
        print(f"EXPERIMENT: FEATURE EXTRACTION")
        for epoch in range(kwargs['TUNE_EPOCHS']):
            print(f"FOLD: {fold_num}, FEATURE EXTRACTION EPOCH: {epoch}")
            # Train all models on each training file; split samples in each file by fold
            for tune_file in list(tune_file_list):
                # Load data and config for this training fold/file
                tune_data, _, ngrid = data_utils.load_fold_data(tune_data_folder, tune_file, fold_num, kwargs['n_folds'])
                ngrid_content = ngrid.get_fill_content()
                print(f"TUNE FILE: {tune_file}, {len(tune_data)} tune samples")
                with open(f"{train_data_folder}train_config.json", "r") as f:
                    config = json.load(f)
                # Construct dataloaders
                base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(tune_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, combine=False, data_subset=kwargs['n_tune_samples'])
                # Train nn models
                for model, loader in zip(nn_model_list, nn_dataloaders):
                    model_utils.set_feature_extraction(model)
                    avg_batch_loss = model_utils.train(model, loader, kwargs['LEARN_RATE'])
        # Save tuned models
        print(f"Fold {fold_num} feature extraction complete, saving model states and metrics...")
        for model in nn_model_list:
            torch.save(model.state_dict(), f"{run_folder}{train_network_folder}models/{model.model_name}_extracted_{fold_num}.pt")
        # Retest each model on the original and generalization networks
        print(f"Evaluating {run_folder}{train_network_folder} on {train_data_folder}")
        for valid_file in train_file_list:
            print(f"VALIDATE FILE: {valid_file}")
            valid_data, ngrid = data_utils.load_all_data(train_data_folder, valid_file)
            ngrid_content = ngrid.get_fill_content()
            with open(f"{train_data_folder}train_config.json", "r") as f:
                config = json.load(f)
            print(f"Successfully loaded {len(valid_data)} testing samples.")
            # Construct dataloaders for all models
            dataloaders = model_utils.make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, data_subset=kwargs['data_subset'])
            # Test all models
            for model, loader in zip(all_model_list, dataloaders):
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Extract_Train_Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Extract_Train_Preds"].extend(list(preds))
        print(f"Evaluating {run_folder}{train_network_folder} on {test_data_folder}")
        for valid_file in test_file_list:
            print(f"VALIDATE FILE: {valid_file}")
            valid_data, ngrid = data_utils.load_all_data(test_data_folder, valid_file)
            ngrid_content = ngrid.get_fill_content()
            with open(f"{train_data_folder}train_config.json", "r") as f:
                config = json.load(f)
            print(f"Successfully loaded {len(valid_data)} testing samples.")
            # Construct dataloaders for all models
            dataloaders = model_utils.make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, data_subset=kwargs['data_subset'])
            # Test all models
            for model, loader in zip(all_model_list, dataloaders):
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Extract_Test_Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Extract_Test_Preds"].extend(list(preds))

        # Calculate various losses:
        fold_results = {
            "Model_Names": [x.model_name for x in all_model_list],
            "Fold": fold_num,
            "Train_Losses": [],
            "Test_Losses": [],
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
    # )