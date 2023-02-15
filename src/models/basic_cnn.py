import numpy as np
import torch
from torch import nn


import torch.nn as nn
import torch.nn.functional as F

## Model Information Possibilities:
# GTFS schedule data
# GTFS schedule + shape data
# GTFS-RT for current trip
# GTFS-RT for current trip + other ongoing trips

## Target:
# Either shingled or unshingled trip travel times
# Stop-stop or location-stop or location-location travel times

## Try:
# - At this point usable for scheduling or arrival times -
# Incorporate the GTFS timetables + stop locations into a CNN
# Then add in network context
# - At this point usable for just arrival times -
# Then add in real-time data on current trip (speed, previous steps)
# Then add in real-time data on adjacent trips (segment speed, nearby speed, headings)

## Notes:
# Should we embed trip or route ID?
# To shingle or not?

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class BasicNeuralNet(nn.Module):
#     def __init__(self, n_features, embed_dict, HIDDEN_SIZE):
#         super(BasicNeuralNet, self).__init__()
#         self.embed_dict = embed_dict
#         self.embed_cols = [embed_dict[x]['col'] for x in embed_dict.keys()]
#         self.flatten = nn.Flatten()
#         self.embeddingTimeID = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
#         self.embeddingWeekID = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(n_features - len(embed_dict.keys()) + embed_dict['timeID']['embed_dims'] + embed_dict['weekID']['embed_dims'], HIDDEN_SIZE),
#             nn.ReLU(),
#             nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
#             nn.ReLU(),
#             nn.Linear(HIDDEN_SIZE, 1),
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         # Embed categorical variables
#         timeID = x[:,self.embed_dict['timeID']['col']].long()
#         weekID = x[:,self.embed_dict['weekID']['col']].long()
#         # Save continuous variables to concat back
#         all_indices = torch.arange(x.shape[1])
#         except_indices = np.setdiff1d(all_indices, self.embed_cols)
#         continuous_X = x[:,except_indices]
#         # Feed data through the model
#         timeID_emb = self.embeddingTimeID(timeID)
#         weekID_emb = self.embeddingWeekID(weekID)
#         x = torch.cat((continuous_X, timeID_emb, weekID_emb), dim=1)
#         pred = self.linear_relu_stack(x)
#         return pred