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
        self.embed_stack = []
        self.flatten = nn.Flatten()
        for embed_feature in self.embed_dict.keys():
            self.embed_stack.append(
                nn.Embedding(embed_dict[embed_feature]['vocab_size'], embed_dict[embed_feature]['embed_dims'])
            )
        embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features - len(embed_dict.keys()) + embed_total_dims, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=0.50),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        # Embed categorical variables
        embed_feature_inputs = []
        for embed_feature in self.embed_dict.keys():
            embed_feature_inputs.append(x[:,self.embed_dict[embed_feature]['col']].long())
        # Save continuous variables to concat back
        all_indices = torch.arange(x.shape[1])
        except_indices = np.setdiff1d(all_indices, self.embed_cols)
        continuous_X = x[:,except_indices]
        # Feed data through the model
        embed_feature_vals = []
        for i, embed_feature in enumerate(self.embed_dict.keys()):
            embed_feature_vals.append(self.embed_stack[i](embed_feature_inputs[i]))

        # Both types
        embed_X = torch.cat(embed_feature_vals, dim=1)
        x = torch.cat((continuous_X, embed_X), dim=1)
        
        # # Only Embeddings
        # embed_X = torch.cat(embed_feature_vals, dim=1)
        # x = embed_X

        # # Only continuous
        # x = continuous_X

        # Make prediction
        pred = self.linear_relu_stack(x)
        return pred

    def fit_to_data(self, train_dataloader, test_dataloader, LEARN_RATE, EPOCHS):
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
                vpreds = self(vinputs)
                vloss = loss_fn(vpreds, vlabels)
                running_vloss += vloss
            avg_valid_loss = running_vloss / validation_steps
            validation_loss.append(avg_valid_loss.item())

            print(f"LOSS: train {avg_batch_loss} valid {avg_valid_loss}")
            epoch_number += 1

        return training_loss, validation_loss
    
    def predict(self, dataloader, config):
        labels = []
        preds = []
        for i, vdata in enumerate(dataloader):
            vinputs, vlabels = vdata
            vpreds = self(vinputs)
            labels.append(vlabels)
            preds.append(vpreds)
        labels = torch.concat(labels).view(-1).detach().numpy()
        preds = torch.concat(preds).view(-1).detach().numpy()
        labels = data_utils.de_normalize(labels, config['time_mean'], config['time_std'])
        preds = data_utils.de_normalize(preds, config['time_mean'], config['time_std'])
        return labels, preds