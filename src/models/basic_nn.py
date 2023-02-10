import numpy as np
import torch
from torch import nn


class BasicNeuralNet(nn.Module):
    def __init__(self, n_features, embed_dict, HIDDEN_SIZE):
        super(BasicNeuralNet, self).__init__()
        self.embed_dict = embed_dict
        self.embed_cols = [embed_dict[x]['col'] for x in embed_dict.keys()]
        self.flatten = nn.Flatten()
        self.embeddingTimeID = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.embeddingWeekID = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features - len(embed_dict.keys()) + embed_dict['timeID']['embed_dims'] + embed_dict['weekID']['embed_dims'], HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        # Embed categorical variables
        timeID = x[:,self.embed_dict['timeID']['col']].long()
        weekID = x[:,self.embed_dict['weekID']['col']].long()
        # Save continuous variables to concat back
        all_indices = torch.arange(x.shape[1])
        except_indices = np.setdiff1d(all_indices, self.embed_cols)
        continuous_X = x[:,except_indices]
        # Feed data through the model
        timeID_emb = self.embeddingTimeID(timeID)
        weekID_emb = self.embeddingWeekID(weekID)
        x = torch.cat((continuous_X, timeID_emb, weekID_emb), dim=1)
        pred = self.linear_relu_stack(x)
        return pred