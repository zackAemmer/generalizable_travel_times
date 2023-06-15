import numpy as np
import torch

from models import ff, rnn, transformer
from utils import data_utils, data_loader


def train(model, dataloader, LEARN_RATE):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    # Use gradients while training
    model.train()
    running_tloss = 0.0
    # Iterate over all batches per training epoch
    num_batches = len(dataloader)
    for i, data in enumerate(dataloader):
        # Run forward/backward
        optimizer.zero_grad()
        # Handles discrepancies in how data/forward differs between models
        _, _, loss = model.batch_step(data)
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
        num_batches = len(dataloader)
        for i, vdata in enumerate(dataloader):
            vlabels, vpreds, loss = model.batch_step(vdata)
            # Accumulate batch loss
            running_vloss += loss.item()
            # Save predictions and labels
            labels.append(vlabels)
            preds.append(vpreds)
        if sequential_flag:
            labels = data_utils.pad_tensors(labels, 1).cpu().detach().numpy()
            preds = data_utils.pad_tensors(preds, 1).cpu().detach().numpy()
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
        for param in model.feature_extract.parameters():
            param.requires_grad = True
        for param in model.feature_extract_activation.parameters():
            param.requires_grad = True

def make_all_models(hidden_size, batch_size, embed_dict, device):
    # Declare neural network models
    nn_model_list = []
    nn_model_list.append(ff.FF(
        "FF",
        n_features=12,
        hidden_size=hidden_size,
        batch_size=batch_size,
        embed_dict=embed_dict,
        device=device
    ).to(device))
    nn_model_list.append(ff.FF_GRID(
        "FF_NGRID_IND",
        n_features=12,
        n_grid_features=3*3*5*5,
        hidden_size=hidden_size,
        grid_compression_size=8,
        batch_size=batch_size,
        embed_dict=embed_dict,
        device=device
    ).to(device))
    nn_model_list.append(rnn.GRU(
        "GRU",
        n_features=10,
        hidden_size=hidden_size,
        batch_size=batch_size,
        embed_dict=embed_dict,
        device=device
    ).to(device))
    nn_model_list.append(rnn.GRU_GRID(
        "GRU_NGRID_IND",
        n_features=10,
        n_grid_features=3*3*5*5,
        hidden_size=hidden_size,
        grid_compression_size=8,
        batch_size=batch_size,
        embed_dict=embed_dict,
        device=device
    ).to(device))
    nn_model_list.append(transformer.TRSF(
        "TRSF",
        n_features=10,
        hidden_size=hidden_size,
        batch_size=batch_size,
        embed_dict=embed_dict,
        device=device
    ).to(device))
    nn_model_list.append(transformer.TRSF_GRID(
        "TRSF_NGRID_IND",
        n_features=10,
        n_grid_features=3*3*5*5,
        hidden_size=hidden_size,
        grid_compression_size=8,
        batch_size=batch_size,
        embed_dict=embed_dict,
        device=device
    ).to(device))
    nn_model_list.append(transformer.TRSF_GRID_ATTN(
        "TRSF_NGRID_CRS",
        n_features=10,
        n_grid_features=3*3*5*5,
        n_channels=3*3,
        hidden_size=hidden_size,
        grid_compression_size=8,
        batch_size=batch_size,
        embed_dict=embed_dict,
        device=device
    ).to(device))
    return nn_model_list

def make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, combine=True, data_subset=None, holdout_routes=None, keep_only_holdout_routes=False):
    # Subset data for faster evaluation
    if data_subset is not None:
        if data_subset < 1:
            valid_data = np.random.choice(valid_data, int(data_subset*len(valid_data)))
        else:
            valid_data = np.random.choice(valid_data, data_subset)
    # Holdout routes for generalization
    if holdout_routes is not None:
        if keep_only_holdout_routes:
            keep_idx = [sample['route_id'] in holdout_routes for sample in valid_data]
            valid_data = [x for i,x in enumerate(valid_data) if keep_idx[i]]
        else:
            keep_idx = [sample['route_id'] not in holdout_routes for sample in valid_data]
            valid_data = [x for i,x in enumerate(valid_data) if keep_idx[i]]
    # Construct dataloaders for all models
    buffer = 2
    base_dataloaders = []
    nn_dataloaders = []
    base_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
    base_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
    base_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=ngrid_content, is_ngrid=True, buffer=buffer))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=ngrid_content, is_ngrid=True, buffer=buffer))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=ngrid_content, is_ngrid=True, buffer=buffer))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=ngrid_content, is_ngrid=True, buffer=buffer))
    if combine:
        base_dataloaders.extend(nn_dataloaders)
        return base_dataloaders
    else:
        return base_dataloaders, nn_dataloaders