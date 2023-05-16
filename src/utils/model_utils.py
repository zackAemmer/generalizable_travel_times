import numpy as np
import torch

from utils import data_utils


def train(model, dataloader, LEARN_RATE):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    # Use gradients while training
    model.train(True)
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
    # Don't use gradients while testing
    model.train(False)
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