import numpy as np
import torch
from torch import nn

from utils import data_utils, model_utils


class FF(nn.Module):
    def __init__(self, model_name, n_features, hidden_size, batch_size, embed_dict, device):
        super(FF, self).__init__()
        self.model_name = model_name
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed_dict = embed_dict
        self.device = device
        self.loss_fn = torch.nn.HuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        # Feedforward
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features + self.embed_total_dims, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size, 1),
        )
    def forward(self, x):
        x_em = x[0]
        x_ct = x[1]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        # Feed data through the model
        x = torch.cat([x_ct, timeID_embedded, weekID_embedded], dim=1)
        # Make prediction
        pred = self.linear_relu_stack(x)
        return pred.squeeze()
    def batch_step(self, data):
        inputs, labels = data
        inputs[:2] = [i.to(self.device) for i in inputs[:2]]
        labels = labels.to(self.device)
        preds = self(inputs)
        loss = self.loss_fn(preds, labels)
        return labels, preds, loss
    def evaluate(self, test_dataloader, config):
        labels, preds, avg_batch_loss = model_utils.predict(self, test_dataloader)
        labels = data_utils.de_normalize(labels, config['time_mean'], config['time_std'])
        preds = data_utils.de_normalize(preds, config['time_mean'], config['time_std'])
        return labels, preds

class FF_GRID(nn.Module):
    def __init__(self, model_name, n_features, n_grid_features, hidden_size, grid_compression_size, batch_size, embed_dict, device):
        super(FF_GRID, self).__init__()
        self.model_name = model_name
        self.n_features = n_features
        self.n_grid_features = n_grid_features
        self.hidden_size = hidden_size
        self.grid_compression_size = grid_compression_size
        self.batch_size = batch_size
        self.embed_dict = embed_dict
        self.device = device
        self.loss_fn = torch.nn.HuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        # Grid Feedforward
        self.linear_relu_stack_grid = nn.Sequential(
            nn.Linear(self.n_grid_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.grid_compression_size),
            nn.ReLU()
        )
        # Feedforward
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features + self.embed_total_dims + self.grid_compression_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size, 1),
        )
    def forward(self, x):
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        # Feed grid data through model
        x = self.linear_relu_stack_grid(x_gr)
        # Feed data through the model
        x = torch.cat([x, x_ct, timeID_embedded, weekID_embedded], dim=1)
        # Make prediction
        pred = self.linear_relu_stack(x)
        return pred.squeeze()
    def batch_step(self, data):
        inputs, labels = data
        inputs[:3] = [i.to(self.device) for i in inputs[:3]]
        labels = labels.to(self.device)
        preds = self(inputs)
        loss = self.loss_fn(preds, labels)
        return labels, preds, loss
    def evaluate(self, test_dataloader, config):
        labels, preds, avg_batch_loss = model_utils.predict(self, test_dataloader)
        labels = data_utils.de_normalize(labels, config['time_mean'], config['time_std'])
        preds = data_utils.de_normalize(preds, config['time_mean'], config['time_std'])
        return labels, preds