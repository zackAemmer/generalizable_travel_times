from datetime import date, datetime, timedelta
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from multiprocessing import Pool
from random import sample
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyproj
from sklearn import metrics
import torch
from torch import nn


def fit_to_data(model, train_dataloader, valid_dataloader, LEARN_RATE, EPOCHS, config, device, sequential_flag=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

    epoch_number = 0
    training_losses = []
    validation_losses = []

    for epoch in range(EPOCHS):
        print(f'EPOCH: {epoch_number}')

        # Use gradients while training
        model.train(True)
        running_tloss = 0.0

        # Iterate over all batches per-epoch
        num_batches = len(train_dataloader)
        for i, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            inputs, labels = data
            if sequential_flag:
                inputs = [i.to(device) for i in inputs]
                hidden_prev = torch.zeros(1, len(data[1]), model.hidden_size).to(device)
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            # Run forward/backward
            optimizer.zero_grad()
            if sequential_flag:
                preds, hidden_prev = model(inputs, hidden_prev)
                hidden_prev = hidden_prev.detach()
            else:
                preds = model(inputs)
            loss = model.loss_fn(preds, labels)
            loss.backward()
            # Adjust weights, save loss
            optimizer.step()
            running_tloss += loss.item()
        # Save the average batch training loss
        avg_batch_tloss = running_tloss / num_batches
        training_losses.append(avg_batch_tloss)

        # Don't use gradients when evaluating
        model.train(False)
        # Calculate the average batch validation loss
        vlabels, vpreds, avg_batch_vloss = predict(model, valid_dataloader, config, device, sequential_flag=sequential_flag)
        validation_losses.append(avg_batch_vloss)
        print(f"LOSS: train {avg_batch_tloss} valid {avg_batch_vloss}")
        epoch_number += 1
    return training_losses, validation_losses

def predict(model, dataloader, config, device, sequential_flag=False):
    labels = []
    preds = []
    avg_batch_loss = 0.0
    num_batches = len(dataloader)
    for i, vdata in enumerate(dataloader):
        # Move data to device
        vinputs, vlabels = vdata
        if sequential_flag:
            vinputs = [vi.to(device) for vi in vinputs]
            hidden_prev = torch.zeros(1, len(vlabels), model.hidden_size).to(device)
        else:
            vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        # Make predictions
        if sequential_flag:
            vpreds, hidden_prev = model(vinputs, hidden_prev)
        else:
            vpreds = model(vinputs)
        # Accumulate batch loss
        avg_batch_loss += model.loss_fn(vpreds, vlabels).item()
        # Save predictions and labels
        labels.append(vlabels)
        preds.append(vpreds)
    labels = torch.concat(labels).cpu().view(-1).detach().numpy()
    preds = torch.concat(preds).cpu().view(-1).detach().numpy()
    return labels, preds, avg_batch_loss / num_batches