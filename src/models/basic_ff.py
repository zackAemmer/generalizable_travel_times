from datetime import datetime

import numpy as np
import torch
from torch import nn

from database import data_utils


class BasicFeedForward(nn.Module):
    def __init__(self, n_features, embed_dict, HIDDEN_SIZE):
        super(BasicFeedForward, self).__init__()
        self.loss_fn = torch.nn.MSELoss()
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