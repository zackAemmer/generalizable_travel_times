from datetime import datetime

import numpy as np
import torch
from torch import nn

from database import data_utils


class BasicFeedForward(nn.Module):
    def __init__(self, n_features, embed_dict, HIDDEN_SIZE):
        super(BasicFeedForward, self).__init__()
        self.embed_dict = embed_dict
        self.embed_cols = [embed_dict[x]['col'] for x in embed_dict.keys()]
        self.flatten = nn.Flatten()
        embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features - len(embed_dict.keys()) + embed_total_dims, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(HIDDEN_SIZE, 1),
        )
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        self.driverID_em = nn.Embedding(embed_dict['driverID']['vocab_size'], embed_dict['driverID']['embed_dims'])

    def forward(self, x):
        x = self.flatten(x)
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x[:,self.embed_dict['timeID']['col']].int())
        weekID_embedded = self.weekID_em(x[:,self.embed_dict['weekID']['col']].int())
        driverID_embedded = self.driverID_em(x[:,self.embed_dict['driverID']['col']].int())
        # Save continuous variables to concat back
        all_indices = np.arange(x.shape[1])
        continuous_indices = np.setdiff1d(all_indices, self.embed_cols)
        continuous_X = x[:,continuous_indices]
        # Feed data through the model
        x = torch.cat([continuous_X, timeID_embedded, weekID_embedded, driverID_embedded], dim=1)
        # Make prediction
        pred = self.linear_relu_stack(x)
        return pred

    def fit_to_data(self, train_dataloader, test_dataloader, LEARN_RATE, EPOCHS, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARN_RATE)
        loss_fn = torch.nn.MSELoss()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0

        training_loss = []
        validation_loss = []
        training_steps = len(train_dataloader)
        validation_steps = len(test_dataloader)

        for epoch in range(EPOCHS):
            print(f'EPOCH: {epoch_number}')

            # Make sure gradient tracking is on, and do a pass over the data
            self.train(True)
            running_tloss = 0.0
            last_loss = 0.0

            # Iterate over all batches per-epoch
            for i, data in enumerate(train_dataloader):
                # Every data instance is an input + label pair
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Run forward/backward
                optimizer.zero_grad()
                preds = self(inputs)
                loss = loss_fn(preds, labels)
                loss.backward()

                # Adjust weights
                optimizer.step()

                # Gather data and report
                running_tloss += loss.item()

            # We don't need gradients on to do reporting
            self.train(False)

            avg_batch_loss = running_tloss / training_steps
            training_loss.append(avg_batch_loss)

            running_vloss = 0.0
            for i, vdata in enumerate(test_dataloader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                vpreds = self(vinputs)
                vloss = loss_fn(vpreds, vlabels)
                running_vloss += vloss
            avg_valid_loss = running_vloss / validation_steps
            validation_loss.append(avg_valid_loss.item())
            print(f"LOSS: train {avg_batch_loss} valid {avg_valid_loss}")
            epoch_number += 1
        return training_loss, validation_loss
    
    def predict(self, dataloader, config, device):
        labels = []
        preds = []
        for i, vdata in enumerate(dataloader):
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            vpreds = self(vinputs)
            labels.append(vlabels)
            preds.append(vpreds)
        labels = torch.concat(labels).cpu().view(-1).detach().numpy()
        preds = torch.concat(preds).cpu().view(-1).detach().numpy()
        labels = data_utils.de_normalize(labels, config['time_mean'], config['time_std'])
        preds = data_utils.de_normalize(preds, config['time_mean'], config['time_std'])
        return labels, preds