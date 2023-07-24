#!/usr/bin python3


import json
import shutil
import random
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn import metrics

from models import grids
from utils import data_utils, model_utils, data_loader


def run_experiments(run_folder, train_network_folder, test_network_folder, tune_network_folder, **kwargs):
    print("="*30)
    print(f"RUN EXPERIMENTS: '{run_folder}'")
    print(f"TRAINED ON NETWORK: '{train_network_folder}'")
    print(f"TUNE ON NETWORK: '{tune_network_folder}'")
    print(f"TEST ON NETWORK: '{test_network_folder}'")

    NUM_WORKERS=4
    PIN_MEMORY=True

    try:
        shutil.rmtree(f"{run_folder}{train_network_folder}gen_logs/")
    except:
        print("Logs folder not found to remove")

    # Define embedded variables for network models
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
            'batch_size': 128,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout_rate': .2
        },
        'CONV': {
            'batch_size': 128,
            'hidden_size': 64,
            'num_layers': 3,
            'dropout_rate': .1
        },
        'GRU': {
            'batch_size': 128,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout_rate': .05
        },
        'TRSF': {
            'batch_size': 128,
            'hidden_size': 64,
            'num_layers': 3,
            'dropout_rate': .1
        },
        'DEEPTTE': {
            'batch_size': 10
        }
    }

    # Data loading and fold setup
    with open(f"{run_folder}{train_network_folder}deeptte_formatted/train_summary_config.json", "r") as f:
        train_network_config = json.load(f)
    with open(f"{run_folder}{test_network_folder}deeptte_formatted/test_summary_config.json", "r") as f:
        test_network_config = json.load(f)
    with open(f"{run_folder}{tune_network_folder}deeptte_formatted/train_summary_config.json", "r") as f:
        tune_network_config = json.load(f)

    print(f"Building grid on validation data from training network")
    train_network_dataset = data_loader.LoadSliceDataset(f"{run_folder}{train_network_folder}deeptte_formatted/test", train_network_config, holdout_routes=kwargs['holdout_routes'], skip_gtfs=kwargs['skip_gtfs'])
    train_network_ngrid = grids.NGridBetter(train_network_config['grid_bounds'][0],kwargs['grid_s_size'])
    train_network_ngrid.add_grid_content(train_network_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    train_network_ngrid.build_cell_lookup()
    train_network_dataset.grid = train_network_ngrid
    print(f"Building grid on validation data from testing network")
    test_network_dataset = data_loader.LoadSliceDataset(f"{run_folder}{test_network_folder}deeptte_formatted/test", test_network_config, holdout_routes=kwargs['holdout_routes'], skip_gtfs=kwargs['skip_gtfs'])
    test_network_ngrid = grids.NGridBetter(test_network_config['grid_bounds'][0],kwargs['grid_s_size'])
    test_network_ngrid.add_grid_content(test_network_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    test_network_ngrid.build_cell_lookup()
    test_network_dataset.grid = test_network_ngrid
    print(f"Building tune grid on training data from testing network")
    tune_network_dataset = data_loader.LoadSliceDataset(f"{run_folder}{tune_network_folder}deeptte_formatted/train", tune_network_config, holdout_routes=kwargs['holdout_routes'], skip_gtfs=kwargs['skip_gtfs'])
    tune_network_ngrid = grids.NGridBetter(tune_network_config['grid_bounds'][0],kwargs['grid_s_size'])
    tune_network_ngrid.add_grid_content(tune_network_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    tune_network_ngrid.build_cell_lookup()
    tune_network_dataset.grid = tune_network_ngrid
    if not kwargs['skip_gtfs']:
        print(f"Building route holdout grid on validation data from training network")
        holdout_network_dataset = data_loader.LoadSliceDataset(f"{run_folder}{train_network_folder}deeptte_formatted/test", train_network_config, holdout_routes=kwargs['holdout_routes'], keep_only_holdout=True, skip_gtfs=kwargs['skip_gtfs'])
        holdout_network_ngrid = grids.NGridBetter(train_network_config['grid_bounds'][0],kwargs['grid_s_size'])
        holdout_network_ngrid.add_grid_content(train_network_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
        holdout_network_ngrid.build_cell_lookup()
        holdout_network_dataset.grid = holdout_network_ngrid

    run_results = []

    # Run experiments on each fold
    for fold_num in range(kwargs['n_folds']):
        print("="*30)
        print(f"BEGIN FOLD: {fold_num}")

        # Declare models
        if not kwargs['skip_gtfs']:
            model_list = model_utils.make_all_models(hyperparameter_dict, embed_dict, train_network_config, load_weights=True, weight_folder=f"{run_folder}{train_network_folder}models/", fold_num=fold_num)
        else:
            model_list = model_utils.make_all_models_nosch(hyperparameter_dict, embed_dict, train_network_config, load_weights=True, weight_folder=f"{run_folder}{train_network_folder}models/", fold_num=fold_num)
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
                "Tune_Test_Preds":[]
            }

        print(f"EXPERIMENT: SAME NETWORK")
        print(f"Evaluating {run_folder}{train_network_folder} on {train_network_folder}")
        for model in model_list:
            print(f"Network {train_network_folder} Fold {fold_num} Model {model.model_name}")
            if not model.is_nn:
                train_network_dataset.add_grid_features = model.requires_grid
                loader = DataLoader(train_network_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                labels, preds = model.evaluate(loader, train_network_config)
                model_fold_results[model.model_name]["Train_Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Train_Preds"].extend(list(preds))
            else:
                train_network_dataset.add_grid_features = model.requires_grid
                loader = DataLoader(train_network_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                trainer = pl.Trainer(
                    limit_predict_batches=.20,
                    logger=CSVLogger(save_dir=f"{run_folder}{train_network_folder}gen_logs/", name=f"{model.model_name}_SAME"),
                )
                preds_and_labels = trainer.predict(model=model, dataloaders=loader)
                preds = np.concatenate([p[0] for p in preds_and_labels])
                labels = np.concatenate([l[1] for l in preds_and_labels])
                model_fold_results[model.model_name]["Train_Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Train_Preds"].extend(list(preds))

        print(f"EXPERIMENT: DIFFERENT NETWORK")
        print(f"Evaluating {run_folder}{train_network_folder} on {test_network_folder}")
        for model in model_list:
            print(f"Network {train_network_folder} Fold {fold_num} Model {model.model_name}")
            if not model.is_nn:
                test_network_dataset.add_grid_features = model.requires_grid
                loader = DataLoader(test_network_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                labels, preds = model.evaluate(loader, train_network_config)
                model_fold_results[model.model_name]["Test_Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Test_Preds"].extend(list(preds))
            else:
                test_network_dataset.add_grid_features = model.requires_grid
                loader = DataLoader(test_network_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                trainer = pl.Trainer(
                    limit_predict_batches=.20,
                    logger=CSVLogger(save_dir=f"{run_folder}{train_network_folder}gen_logs/", name=f"{model.model_name}_DIFF"),
                )
                preds_and_labels = trainer.predict(model=model, dataloaders=loader)
                preds = np.concatenate([p[0] for p in preds_and_labels])
                labels = np.concatenate([l[1] for l in preds_and_labels])
                model_fold_results[model.model_name]["Test_Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Test_Preds"].extend(list(preds))

        if not kwargs['skip_gtfs']:
            print(f"EXPERIMENT: HOLDOUT ROUTES")
            print(f"Evaluating {run_folder}{train_network_folder} on holdout routes from {train_network_folder}")
            for model in model_list:
                print(f"Network {train_network_folder} Fold {fold_num} Model {model.model_name}")
                if not model.is_nn:
                    holdout_network_dataset.add_grid_features = model.requires_grid
                    loader = DataLoader(holdout_network_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                    labels, preds = model.evaluate(loader, train_network_config)
                    model_fold_results[model.model_name]["Holdout_Labels"].extend(list(labels))
                    model_fold_results[model.model_name]["Holdout_Preds"].extend(list(preds))
                else:
                    holdout_network_dataset.add_grid_features = model.requires_grid
                    loader = DataLoader(holdout_network_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                    trainer = pl.Trainer(
                        limit_predict_batches=.20,
                        logger=CSVLogger(save_dir=f"{run_folder}{train_network_folder}gen_logs/", name=f"{model.model_name}_HOLD"),
                    )
                    preds_and_labels = trainer.predict(model=model, dataloaders=loader)
                    preds = np.concatenate([p[0] for p in preds_and_labels])
                    labels = np.concatenate([l[1] for l in preds_and_labels])
                    model_fold_results[model.model_name]["Holdout_Labels"].extend(list(labels))
                    model_fold_results[model.model_name]["Holdout_Preds"].extend(list(preds))

            print(f"EXPERIMENT: FINE TUNING")
            # Re-declare models with original weights
            model_list = model_utils.make_all_models(hyperparameter_dict, embed_dict, train_network_config, load_weights=True, weight_folder=f"{run_folder}{train_network_folder}models/", fold_num=fold_num)
            # Tune model on tuning network
            for model in model_list:
                print(f"Network {train_network_folder} Fold {fold_num} Model {model.model_name}")
                if not model.is_nn:
                    tune_network_dataset.add_grid_features = model.requires_grid
                    loader = DataLoader(tune_network_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                    model.train(loader, train_network_config)
                    print(f"Evaluating model {model.model_name} on {train_network_folder}")
                    train_network_dataset.add_grid_features = model.requires_grid
                    loader = DataLoader(train_network_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                    labels, preds = model.evaluate(loader, train_network_config)
                    model_fold_results[model.model_name]["Tune_Train_Labels"].extend(list(labels))
                    model_fold_results[model.model_name]["Tune_Train_Preds"].extend(list(preds))
                    print(f"Evaluating model {model.model_name} on {test_network_folder}")
                    test_network_dataset.add_grid_features = model.requires_grid
                    loader = DataLoader(test_network_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                    labels, preds = model.evaluate(loader, test_network_config)
                    model_fold_results[model.model_name]["Tune_Test_Labels"].extend(list(labels))
                    model_fold_results[model.model_name]["Tune_Test_Preds"].extend(list(preds))
                else:
                    tune_network_dataset.add_grid_features = model.requires_grid
                    loader = DataLoader(tune_network_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                    trainer = pl.Trainer(
                        limit_predict_batches=.20,
                        check_val_every_n_epoch=2,
                        max_epochs=kwargs['TUNE_EPOCHS'],
                        min_epochs=1,
                        logger=CSVLogger(save_dir=f"{run_folder}{train_network_folder}gen_logs/", name=f"{model.model_name}_TUNE"),
                        callbacks=[EarlyStopping(monitor=f"{model.model_name}_valid_loss", min_delta=.001, patience=3)]
                    )
                    trainer.fit(model=model, train_dataloaders=loader)
                    print(f"Evaluating model {model.model_name} on {train_network_folder}")
                    train_network_dataset.add_grid_features = model.requires_grid
                    loader = DataLoader(train_network_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                    preds_and_labels = trainer.predict(model=model, dataloaders=loader)
                    preds = np.concatenate([p[0] for p in preds_and_labels])
                    labels = np.concatenate([l[1] for l in preds_and_labels])
                    model_fold_results[model.model_name]["Tune_Train_Labels"].extend(list(labels))
                    model_fold_results[model.model_name]["Tune_Train_Preds"].extend(list(preds))
                    print(f"Evaluating model {model.model_name} on {test_network_folder}")
                    test_network_dataset.add_grid_features = model.requires_grid
                    loader = DataLoader(test_network_dataset, collate_fn=model.collate_fn, batch_size=model.batch_size, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                    preds_and_labels = trainer.predict(model=model, dataloaders=loader)
                    preds = np.concatenate([p[0] for p in preds_and_labels])
                    labels = np.concatenate([l[1] for l in preds_and_labels])
                    model_fold_results[model.model_name]["Tune_Test_Labels"].extend(list(labels))
                    model_fold_results[model.model_name]["Tune_Test_Preds"].extend(list(preds))
            # Save tuned models
            print(f"Fold {fold_num} fine tuning complete, saving model states and metrics...")
            for model in model_list:
                if model.is_nn:
                    continue
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
            if not kwargs['skip_gtfs']:
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

        # Save fold
        run_results.append(fold_results)

    # Save run results
    data_utils.write_pkl(run_results, f"{run_folder}{train_network_folder}model_generalization_results.pkl")
    print(f"EXPERIMENTS COMPLETED '{run_folder}{train_network_folder}'")


if __name__=="__main__":
    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    # # DEBUG
    # run_experiments(
    #     run_folder="./results/debug/",
    #     train_network_folder="kcm/",
    #     test_network_folder="atb/",
    #     tune_network_folder="atb/",
    #     TUNE_EPOCHS=2,
    #     grid_s_size=500,
    #     n_tune_samples=100,
    #     n_folds=2,
    #     holdout_routes=[100252,100139,102581,100341,102720],
    #     skip_gtfs=False
    # )
    # run_experiments(
    #     run_folder="./results/debug/",
    #     train_network_folder="atb/",
    #     test_network_folder="kcm/",
    #     tune_network_folder="kcm/",
    #     TUNE_EPOCHS=2,
    #     grid_s_size=500,
    #     n_tune_samples=100,
    #     n_folds=2,
    #     holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"],
    #     skip_gtfs=False
    # )

    # FULL RUN
    run_experiments(
        run_folder="./results/full_run/",
        train_network_folder="kcm/",
        test_network_folder="atb/",
        tune_network_folder="atb/",
        TUNE_EPOCHS=5,
        grid_s_size=500,
        n_tune_samples=100,
        n_folds=5,
        holdout_routes=[100252,100139,102581,100341,102720],
        skip_gtfs=False
    )
    run_experiments(
        run_folder="./results/full_run/",
        train_network_folder="atb/",
        test_network_folder="kcm/",
        tune_network_folder="kcm/",
        TUNE_EPOCHS=5,
        grid_s_size=500,
        n_tune_samples=100,
        n_folds=5,
        holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"],
        skip_gtfs=False
    )