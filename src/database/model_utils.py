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

from database import data_utils


def fit_to_data(model, train_dataloader, test_dataloader, LEARN_RATE, EPOCHS, device, sequential_flag=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    loss_fn = torch.nn.MSELoss()

    epoch_number = 0

    training_loss = []
    validation_loss = []
    training_steps = len(train_dataloader)
    validation_steps = len(test_dataloader)

    for epoch in range(EPOCHS):
        print(f'EPOCH: {epoch_number}')
        model.train(True)
        running_tloss = 0.0
        last_loss = 0.0

        # Iterate over all batches per-epoch
        for i, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            inputs, labels = data
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(device)
            labels = labels.to(device)
            
            # If model is sequence, requires initial hidden outputs
            if sequential_flag:
                hidden_prev = torch.zeros(1, len(data[1]), model.hidden_size).to(device)

            # Run forward/backward
            optimizer.zero_grad()
            if sequential_flag:
                preds, hidden_prev = model(inputs, hidden_prev)
                hidden_prev = hidden_prev.detach()
            else:
                preds = model(inputs)
            loss = loss_fn(preds, labels)
            loss.backward()

            # Adjust weights
            optimizer.step()

            # Gather data and report
            running_tloss += loss.item()

        # We don't need gradients on to do reporting
        model.train(False)

        avg_batch_loss = running_tloss / training_steps
        training_loss.append(avg_batch_loss)

        running_vloss = 0.0
        for i, vdata in enumerate(test_dataloader):
            vinputs, vlabels = vdata
            if sequential_flag:
                hidden_prev = torch.zeros(1, len(vlabels), model.hidden_size).to(device)
            for i in range(len(vinputs)):
                vinputs[i] = vinputs[i].to(device)
            vlabels = vlabels.to(device)
            vpreds = model(vinputs, hidden_prev)
            if len(vpreds) > 1:
                vpreds = vpreds[0]
            vloss = loss_fn(vpreds, vlabels)
            running_vloss += vloss
        avg_valid_loss = running_vloss / validation_steps
        validation_loss.append(avg_valid_loss.item())
        print(f"LOSS: train {avg_batch_loss} valid {avg_valid_loss}")
        epoch_number += 1
    return training_loss, validation_loss

def predict(model, dataloader, config, device, sequential_flag=False):
    labels = []
    preds = []
    for i, vdata in enumerate(dataloader):
        vinputs, vlabels = vdata
        if sequential_flag:
            hidden_prev = torch.zeros(1, len(vlabels), model.hidden_size).to(device)
        for i in range(len(vinputs)):
            vinputs[i] = vinputs[i].to(device)
        vlabels = vlabels.to(device)
        vpreds = model(vinputs, hidden_prev)
        if len(vpreds) > 1:
            vpreds = vpreds[0]
        labels.append(vlabels)
        preds.append(vpreds)
    labels = torch.concat(labels).cpu().view(-1).detach().numpy()
    preds = torch.concat(preds).cpu().view(-1).detach().numpy()
    labels = data_utils.de_normalize(labels, config['time_mean'], config['time_std'])
    preds = data_utils.de_normalize(preds, config['time_mean'], config['time_std'])
    return labels, preds