import numpy as np
import torch

from utils import data_utils


def train_model(model, train_dataloader, valid_dataloader, LEARN_RATE, EPOCHS, sequential_flag=False):
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
            # Run forward/backward
            optimizer.zero_grad()
            # Handles discrepancies in how data/forward differs between models
            _, _, loss = model.batch_step(data)
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
        _, _, avg_batch_vloss = predict(model, valid_dataloader, sequential_flag=sequential_flag)
        validation_losses.append(avg_batch_vloss)
        print(f"LOSS: train {avg_batch_tloss} valid {avg_batch_vloss}")
    return training_losses, validation_losses

def predict(model, dataloader, sequential_flag=False):
    labels = []
    preds = []
    avg_batch_loss = 0.0
    num_batches = len(dataloader)
    for i, vdata in enumerate(dataloader):
        vlabels, vpreds, loss = model.batch_step(vdata)
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