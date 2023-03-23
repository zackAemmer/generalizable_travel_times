import numpy as np
import torch

from utils import data_utils


def fit_to_data(model, train_dataloader, valid_dataloader, LEARN_RATE, EPOCHS, config, device, sequential_flag=False, grid_flag=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    training_losses = []
    validation_losses = []

    for epoch in range(EPOCHS):
        print(f'EPOCH: {epoch}')

        # Use gradients while training
        model.train(True)
        running_tloss = 0.0

        # Iterate over all batches per training epoch
        num_batches = len(train_dataloader)
        for i, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs[:2] = [i.to(device) for i in inputs[:2]]
            labels = labels.to(device)
            # Run forward/backward
            optimizer.zero_grad()
            if sequential_flag:
                hidden_prev = torch.zeros(1, len(data[1]), model.hidden_size).to(device)
                preds, hidden_prev = model(inputs, hidden_prev)
                hidden_prev = hidden_prev.detach()
                mask = data_utils.create_tensor_mask(inputs[2]).to(device)
                loss = model.loss_fn(preds, labels, mask)
            else:
                if grid_flag:
                    inputs[:3] = [i.to(device) for i in inputs[:3]]
                    preds = model(inputs)
                    loss = model.loss_fn(preds, labels)
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
        vlabels, vpreds, avg_batch_vloss = predict(model, valid_dataloader, device, sequential_flag=sequential_flag, grid_flag=grid_flag)
        validation_losses.append(avg_batch_vloss)
        print(f"LOSS: train {avg_batch_tloss} valid {avg_batch_vloss}")
    return training_losses, validation_losses

def predict(model, dataloader, device, sequential_flag=False, grid_flag=False):
    labels = []
    preds = []
    avg_batch_loss = 0.0
    num_batches = len(dataloader)
    for i, vdata in enumerate(dataloader):
        # Move data to device
        vinputs, vlabels = vdata
        vinputs[:2] = [vi.to(device) for vi in vinputs[:2]]
        vlabels = vlabels.to(device)
        # Make predictions
        if sequential_flag:
            hidden_prev = torch.zeros(1, len(vlabels), model.hidden_size).to(device)
            vpreds, hidden_prev = model(vinputs, hidden_prev)
            hidden_prev = hidden_prev.detach()
            vmask = data_utils.create_tensor_mask(vinputs[2]).to(device)
            loss = model.loss_fn(vpreds, vlabels, vmask)
        else:
            if grid_flag:
                vinputs[:3] = [vi.to(device) for vi in vinputs[:3]]
                vpreds = model(vinputs)
                loss = model.loss_fn(vpreds, vlabels)
            else:
                vpreds = model(vinputs)
                loss = model.loss_fn(vpreds, vlabels)
        # Accumulate batch loss
        avg_batch_loss += loss.item()
        # Save predictions and labels
        labels.append(vlabels)
        preds.append(vpreds)
    if sequential_flag:
        labels = data_utils.pad_tensors(labels, 1).cpu().detach().numpy()
        preds = data_utils.pad_tensors(preds, 1).cpu().detach().numpy()
    else:
        labels = torch.concat(labels).cpu().detach().numpy()
        preds = torch.concat(preds).cpu().detach().numpy()
    return labels, preds, avg_batch_loss / num_batches