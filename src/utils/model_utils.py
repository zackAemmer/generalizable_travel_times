import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models import ff, rnn, transformer
from models.deeptte import DeepTTE
from utils import data_utils, data_loader


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
        for vdata in dataloader:
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

def make_all_models(hidden_size, batch_size, embed_dict, device, config):
    # Declare neural network models
    nn_model_list = []
    # nn_model_list.append(ff.FF(
    #     "FF",
    #     n_features=12,
    #     hidden_size=hidden_size,
    #     batch_size=batch_size,
    #     embed_dict=embed_dict,
    #     collate_fn=data_loader.basic_collate,
    #     device=device
    # ).to(device))
    # nn_model_list.append(ff.FF_GRID(
    #     "FF_NGRID_IND",
    #     n_features=12,
    #     n_grid_features=3*3*5*5,
    #     hidden_size=hidden_size,
    #     grid_compression_size=8,
    #     batch_size=batch_size,
    #     embed_dict=embed_dict,
    #     collate_fn=data_loader.basic_grid_collate,
    #     device=device
    # ).to(device))
    # nn_model_list.append(rnn.GRU(
    #     "GRU",
    #     n_features=10,
    #     hidden_size=hidden_size,
    #     batch_size=batch_size,
    #     embed_dict=embed_dict,
    #     collate_fn=data_loader.sequential_collate,
    #     device=device
    # ).to(device))
    # nn_model_list.append(rnn.GRU_GRID(
    #     "GRU_NGRID_IND",
    #     n_features=10,
    #     n_grid_features=3*3*5*5,
    #     hidden_size=hidden_size,
    #     grid_compression_size=8,
    #     batch_size=batch_size,
    #     embed_dict=embed_dict,
    #     collate_fn=data_loader.sequential_grid_collate,
    #     device=device
    # ).to(device))
    # nn_model_list.append(transformer.TRSF(
    #     "TRSF",
    #     n_features=10,
    #     hidden_size=hidden_size,
    #     batch_size=batch_size,
    #     embed_dict=embed_dict,
    #     collate_fn=data_loader.sequential_collate,
    #     device=device
    # ).to(device))
    # nn_model_list.append(transformer.TRSF_GRID(
    #     "TRSF_NGRID_IND",
    #     n_features=10,
    #     n_grid_features=3*3*5*5,
    #     hidden_size=hidden_size,
    #     grid_compression_size=8,
    #     batch_size=batch_size,
    #     embed_dict=embed_dict,
    #     collate_fn=data_loader.sequential_grid_collate,
    #     device=device
    # ).to(device))
    # nn_model_list.append(transformer.TRSF_GRID_ATTN(
    #     "TRSF_NGRID_CRS",
    #     n_features=10,
    #     n_grid_features=3*3*5*5,
    #     n_channels=3*3,
    #     hidden_size=hidden_size,
    #     grid_compression_size=8,
    #     batch_size=batch_size,
    #     embed_dict=embed_dict,
    #     collate_fn=data_loader.sequential_grid_collate,
    #     device=device
    # ).to(device))
    nn_model_list.append(DeepTTE.Net(
        "DEEP_TTE",
        collate_fn=data_loader.deeptte_collate,
        device=device,
        config=config
    ).to(device))
    return nn_model_list

# def make_all_dataloaders(dataset, sampler, batch_size, num_workers):
#     if num_workers > 0:
#         pin_memory=True
#     else:
#         pin_memory=False
#     base_dataloaders = []
#     # Construct base datalaoders
#     base_dataloaders.append(DataLoader(dataset, sampler=sampler, collate_fn=data_loader.basic_collate, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers))
#     base_dataloaders.append(DataLoader(dataset, sampler=sampler, collate_fn=data_loader.basic_collate, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers))
#     base_dataloaders.append(DataLoader(dataset, sampler=sampler, collate_fn=data_loader.sequential_collate, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers))
#     # Construct nn dataloaders
#     nn_dataloaders = []
#     nn_dataloaders.append(DataLoader(dataset, sampler=sampler, collate_fn=data_loader.basic_collate, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers))
#     nn_dataloaders.append(DataLoader(dataset, sampler=sampler, collate_fn=data_loader.sequential_collate, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers))
#     nn_dataloaders.append(DataLoader(dataset, sampler=sampler, collate_fn=data_loader.sequential_collate, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers))
#     return base_dataloaders, nn_dataloaders