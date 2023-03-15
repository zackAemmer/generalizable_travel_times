import numpy as np
import torch

from utils import data_utils


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

        # Iterate over all batches per training epoch
        num_batches = len(train_dataloader)
        for i, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = [i.to(device) for i in inputs]
            labels = labels.to(device)
            # Run forward/backward
            optimizer.zero_grad()
            if sequential_flag:
                hidden_prev = torch.zeros(1, len(data[1]), model.hidden_size).to(device)
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
        vlabels, vpreds, avg_batch_vloss = predict(model, valid_dataloader, device, sequential_flag=sequential_flag)
        validation_losses.append(avg_batch_vloss)
        print(f"LOSS: train {avg_batch_tloss} valid {avg_batch_vloss}")
        epoch_number += 1
    return training_losses, validation_losses

def predict(model, dataloader, device, sequential_flag=False):
    labels = []
    preds = []
    avg_batch_loss = 0.0
    num_batches = len(dataloader)
    for i, vdata in enumerate(dataloader):
        # Move data to device
        vinputs, vlabels = vdata
        vinputs = [vi.to(device) for vi in vinputs]
        vlabels = vlabels.to(device)
        # Make predictions
        if sequential_flag:
            hidden_prev = torch.zeros(1, len(vlabels), model.hidden_size).to(device)
            vpreds, hidden_prev = model(vinputs, hidden_prev)
            hidden_prev = hidden_prev.detach()
        else:
            vpreds = model(vinputs)
        # Accumulate batch loss
        avg_batch_loss += model.loss_fn(vpreds, vlabels).item()
        # Save predictions and labels
        labels.append(vlabels)
        preds.append(vpreds)
    labels = torch.concat(labels).cpu().detach().numpy()
    preds = torch.concat(preds).cpu().detach().numpy()
    return labels, preds, avg_batch_loss / num_batches

def convert_speeds_to_tts(speeds, dataloader, mask, config):
    # Calculate travel time given a dataloader with sequential distances and speed predictions
    data, labels = dataloader.dataset.tensors
    dists = [x[0][:,2][mask[i]].numpy() for i,x in enumerate(data)]
    dists = [data_utils.de_normalize(x, config['dist_calc_km_mean'], config['dist_calc_km_std']) for x in dists]
    # Speeds can be very close, but not exactly zero to avoid errors
    masked_speeds = [np.maximum(0.001, x[mask[i]]) for i,x in enumerate(speeds)]
    # Get travel time for every step of every sequence, sum to get shingle tt
    res = [i*1000.0/j for i,j in zip(dists,masked_speeds)]
    res = np.array([np.sum(x) for x in res], dtype='float32')
    return res