import numpy as np
import torch
from torch import nn


class BasicFeedForward(nn.Module):
    def __init__(self, n_features, embed_dict, HIDDEN_SIZE):
        super(BasicFeedForward, self).__init__()
        self.loss_fn = torch.nn.MSELoss()
        self.embed_dict = embed_dict
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        self.driverID_em = nn.Embedding(embed_dict['driverID']['vocab_size'], embed_dict['driverID']['embed_dims'])
        self.tripID_em = nn.Embedding(embed_dict['tripID']['vocab_size'], embed_dict['tripID']['embed_dims'])
        # Feedforward
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features + self.embed_total_dims, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        x_ct = x[0]
        x_em = x[1]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        driverID_embedded = self.driverID_em(x_em[:,2])
        tripID_embedded = self.tripID_em(x_em[:,3])
        # Feed data through the model
        x = torch.cat([x_ct, timeID_embedded, weekID_embedded, driverID_embedded, tripID_embedded], dim=1)
        # Make prediction
        pred = self.linear_relu_stack(x)
        return pred.squeeze()