import numpy as np
import torch

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

def make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, grid_content, ngrid, combine=True, data_subset=None):
    # Subset data if applicable
    if data_subset is not None:
        valid_data = np.random.choice(valid_data, int(data_subset*len(valid_data)))
    # Construct dataloaders for all models
    buffer = 1
    base_dataloaders = []
    nn_dataloaders = []
    base_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
    base_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
    base_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=buffer))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=buffer))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=buffer))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=buffer))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=buffer))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=buffer))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=buffer))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=buffer))
    nn_dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=buffer))
    if combine:
        base_dataloaders.extend(nn_dataloaders)
        return base_dataloaders
    else:
        return base_dataloaders, nn_dataloaders