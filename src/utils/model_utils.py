import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
import os

from models import ff, conv, rnn, transformer, avg_speed, schedule, persistent
from models.deeptte import DeepTTE
from utils import data_loader
from utils import data_utils


def train(model, dataloader, optimizer):
    # Use gradients while training
    model.train()
    running_tloss = 0.0
    # Iterate over all batches per training epoch
    num_batches = len(dataloader)
    for data in dataloader:
        # Run forward/backward
        optimizer.zero_grad()
        # Handles discrepancies in how data/forward differs between models
        loss = model.batch_step(data)[2]
        loss.backward()
        # Adjust weights, save loss
        optimizer.step()
        running_tloss += loss.item()
    avg_batch_tloss = running_tloss / num_batches
    return avg_batch_tloss

def predict(model, dataloader, sequential_flag=False):
    # Don't use dropout etc.
    model.eval()
    # Don't track gradients
    with torch.no_grad():
        running_vloss = 0.0
        labels = []
        preds = []
        seq_lens = []
        num_batches = len(dataloader)
        for vdata in dataloader:
            if sequential_flag:
                vlabels, vpreds, loss, vseq_lens = model.batch_step(vdata)
            else:
                vlabels, vpreds, loss = model.batch_step(vdata)
            # Handle batch of 1
            if vpreds.dim()==0:
                vpreds = torch.unsqueeze(vpreds, 0)
            # Accumulate batch loss
            running_vloss += loss.item()
            # Save predictions and labels
            labels.append(vlabels)
            preds.append(vpreds)
            if sequential_flag:
                seq_lens.append(vseq_lens)
        if sequential_flag:
            labels = data_utils.pad_tensors(labels, 1).cpu().detach().numpy()
            preds = data_utils.pad_tensors(preds, 1).cpu().detach().numpy()
            avg_batch_loss = running_vloss / num_batches
            return labels, preds, avg_batch_loss, seq_lens
        else:
            labels = torch.concat(labels).cpu().detach().numpy()
            preds = torch.concat(preds).cpu().detach().numpy()
            avg_batch_loss = running_vloss / num_batches
            return labels, preds, avg_batch_loss

def set_feature_extraction(model, feature_extraction=True):
    if feature_extraction==False:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
        # Each model must have a final named feature extraction layer
        if model.model_name=="DEEP_TTE":
            for param in model.entire_estimate.feature_extract.parameters():
                param.requires_grad = True
            for param in model.local_estimate.feature_extract.parameters():
                param.requires_grad = True
        else:
            for param in model.feature_extract.parameters():
                param.requires_grad = True

def random_param_search(hyperparameter_sample_dict, model_names):
    # Keep list of hyperparam dicts; each is randomly sampled from the given; repeat dict for each model
    set_of_random_dicts = []
    for i in range(hyperparameter_sample_dict['n_param_samples']):
        all_model_dict = {}
        random_dict = {}
        for key in list(hyperparameter_sample_dict.keys()):
            random_dict[key] = np.random.choice(hyperparameter_sample_dict[key],1)[0]
        for mname in model_names:
            all_model_dict[mname] = random_dict
        set_of_random_dicts.append(all_model_dict)
    return set_of_random_dicts

def make_all_models(hyperparameter_dict, embed_dict, config, load_weights=False, weight_folder=None, fold_num=None):
    # Declare base models
    base_model_list = []
    base_model_list.append(avg_speed.AvgHourlySpeedModel("AVG"))
    base_model_list.append(schedule.TimeTableModel("SCH"))
    base_model_list.append(persistent.PersistentTimeSeqModel("PER_TIM"))
    # Declare neural network models
    nn_model_list = []
    nn_model_list.append(ff.FF_L(
        "FF",
        n_features=14,
        hyperparameter_dict=hyperparameter_dict['FF'],
        embed_dict=embed_dict,
        collate_fn=data_loader.basic_collate,
        config=config
    ))
    nn_model_list.append(ff.FF_GRID_L(
        "FF_NGRID_IND",
        n_features=14,
        n_grid_features=3*3*5*5,
        grid_compression_size=8,
        hyperparameter_dict=hyperparameter_dict['FF'],
        embed_dict=embed_dict,
        collate_fn=data_loader.basic_grid_collate,
        config=config
    ))
    nn_model_list.append(conv.CONV_L(
        "CONV",
        n_features=10,
        hyperparameter_dict=hyperparameter_dict['CONV'],
        embed_dict=embed_dict,
        collate_fn=data_loader.sequential_collate,
        config=config
    ))
    nn_model_list.append(conv.CONV_GRID_L(
        "CONV_NGRID_IND",
        n_features=10,
        n_grid_features=3*3*5*5,
        grid_compression_size=8,
        hyperparameter_dict=hyperparameter_dict['CONV'],
        embed_dict=embed_dict,
        collate_fn=data_loader.sequential_grid_collate,
        config=config
    ))
    nn_model_list.append(rnn.GRU_L(
        "GRU",
        n_features=10,
        hyperparameter_dict=hyperparameter_dict['GRU'],
        embed_dict=embed_dict,
        collate_fn=data_loader.sequential_collate,
        config=config
    ))
    nn_model_list.append(rnn.GRU_GRID_L(
        "GRU_NGRID_IND",
        n_features=10,
        n_grid_features=3*3*5*5,
        grid_compression_size=8,
        hyperparameter_dict=hyperparameter_dict['GRU'],
        embed_dict=embed_dict,
        collate_fn=data_loader.sequential_grid_collate,
        config=config
    ))
    nn_model_list.append(transformer.TRSF_L(
        "TRSF",
        n_features=10,
        hyperparameter_dict=hyperparameter_dict['TRSF'],
        embed_dict=embed_dict,
        collate_fn=data_loader.sequential_collate,
        config=config
    ))
    nn_model_list.append(transformer.TRSF_GRID_L(
        "TRSF_NGRID_IND",
        n_features=10,
        n_grid_features=3*3*5*5,
        grid_compression_size=8,
        hyperparameter_dict=hyperparameter_dict['TRSF'],
        embed_dict=embed_dict,
        collate_fn=data_loader.sequential_grid_collate,
        config=config
    ))
    nn_model_list.append(DeepTTE.Net(
        "DEEP_TTE",
        hyperparameter_dict=hyperparameter_dict['DEEPTTE'],
        collate_fn=data_loader.deeptte_collate,
        config=config
    ))
    # Load weights if applicable
    if load_weights:
        base_model_list = []
        for b in base_model_list:
            b = data_utils.load_pkl(f"{weight_folder}{b.name}_{fold_num}.pkl")
        for m in nn_model_list:
            last_ckpt = os.listdir(f"{weight_folder}/../logs/{m.model_name}/version_{fold_num}/")
            m = m.load_from_checkpoint(f"{weight_folder}/../logs/{m.model_name}/version_{fold_num}/{last_ckpt}.ckpt").eval()
            # m = m.load_state_dict(torch.load(f"{weight_folder}{m.model_name}_{fold_num}.pt"))    # Combine all models
    base_model_list.extend(nn_model_list)
    return base_model_list

def make_all_models_nosch(hyperparameter_dict, embed_dict, config, load_weights=False, weight_folder=None, fold_num=None):
    # Declare base models
    base_model_list = []
    base_model_list.append(avg_speed.AvgHourlySpeedModel("AVG"))
    base_model_list.append(persistent.PersistentTimeSeqModel("PER_TIM"))
    # Declare neural network models
    nn_model_list = []
    nn_model_list.append(ff.FF_L(
        "FF",
        n_features=8,
        hyperparameter_dict=hyperparameter_dict['FF'],
        embed_dict=embed_dict,
        collate_fn=data_loader.basic_collate_nosch,
        config=config
    ))
    nn_model_list.append(conv.CONV_L(
        "CONV",
        n_features=4,
        hyperparameter_dict=hyperparameter_dict['CONV'],
        embed_dict=embed_dict,
        collate_fn=data_loader.sequential_collate_nosch,
        config=config
    ))
    nn_model_list.append(rnn.GRU_L(
        "GRU",
        n_features=4,
        hyperparameter_dict=hyperparameter_dict['GRU'],
        embed_dict=embed_dict,
        collate_fn=data_loader.sequential_collate_nosch,
        config=config
    ))
    nn_model_list.append(transformer.TRSF_L(
        "TRSF",
        n_features=4,
        hyperparameter_dict=hyperparameter_dict['TRSF'],
        embed_dict=embed_dict,
        collate_fn=data_loader.sequential_collate_nosch,
        config=config
    ))
    # nn_model_list.append(DeepTTE.Net(
    #     "DEEP_TTE",
    #     hyperparameter_dict=hyperparameter_dict['DEEPTTE'],
    #     collate_fn=data_loader.deeptte_collate_nosch,
    #     config=config
    # ))
    # Load weights if applicable
    if load_weights:
        base_model_list = []
        for b in base_model_list:
            b = data_utils.load_pkl(f"{weight_folder}{b.name}_{fold_num}.pkl")
        for m in nn_model_list:
            last_ckpt = os.listdir(f"{weight_folder}/../logs/{m.model_name}/version_{fold_num}/")
            m = m.load_from_checkpoint(f"{weight_folder}/../logs/{m.model_name}/version_{fold_num}/{last_ckpt}.ckpt").eval()
            # m = m.load_state_dict(torch.load(f"{weight_folder}{m.model_name}_{fold_num}.pt"))
    # Combine all models
    base_model_list.extend(nn_model_list)
    return base_model_list

def make_param_search_models(hyperparameter_dict, embed_dict, config):
    # Declare base models
    base_model_list = []
    # Declare neural network models
    nn_model_list = []
    for sample_num, sample_dict in enumerate(hyperparameter_dict):
        nn_model_list.append(ff.FF_L(
            f"FF_{sample_num}",
            n_features=14,
            hyperparameter_dict=sample_dict['FF'],
            embed_dict=embed_dict,
            collate_fn=data_loader.basic_collate,
            config=config
        ))
        nn_model_list.append(conv.CONV_L(
            f"CONV_{sample_num}",
            n_features=10,
            hyperparameter_dict=sample_dict['CONV'],
            embed_dict=embed_dict,
            collate_fn=data_loader.sequential_collate,
            config=config
        ))
        nn_model_list.append(rnn.GRU_L(
            f"GRU_{sample_num}",
            n_features=10,
            hyperparameter_dict=sample_dict['GRU'],
            embed_dict=embed_dict,
            collate_fn=data_loader.sequential_collate,
            config=config
        ))
        nn_model_list.append(transformer.TRSF_L(
            f"TRSF_{sample_num}",
            n_features=10,
            hyperparameter_dict=sample_dict['TRSF'],
            embed_dict=embed_dict,
            collate_fn=data_loader.sequential_collate,
            config=config
        ))
    # Combine all models
    base_model_list.extend(nn_model_list)
    return base_model_list