import numpy as np
import torch
from torch import nn

from utils import data_utils, model_utils


class FF(nn.Module):
    def __init__(self, model_name, n_features, hidden_size, embed_dict, device):
        super(FF, self).__init__()
        self.model_name = model_name
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.embed_dict = embed_dict
        self.device = device
        self.loss_fn = torch.nn.MSELoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        # self.driverID_em = nn.Embedding(embed_dict['driverID']['vocab_size'], embed_dict['driverID']['embed_dims'])
        # self.tripID_em = nn.Embedding(embed_dict['tripID']['vocab_size'], embed_dict['tripID']['embed_dims'])
        # Feedforward
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features + self.embed_total_dims, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_size, 1),
        )
    def forward(self, x):
        x_em = x[0]
        x_ct = x[1]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        # driverID_embedded = self.driverID_em(x_em[:,2])
        # tripID_embedded = self.tripID_em(x_em[:,3])
        # Feed data through the model
        # x = torch.cat([x_ct, timeID_embedded, weekID_embedded, driverID_embedded, tripID_embedded], dim=1)
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
    def fit_to_data(self, train_dataloader, test_dataloader, config, learn_rate, epochs):
        train_losses, test_losses = model_utils.train_model(self, train_dataloader, test_dataloader, learn_rate, epochs)
        labels, preds, avg_loss = model_utils.predict(self, test_dataloader)
        labels = data_utils.de_normalize(labels, config['time_mean'], config['time_std'])
        preds = data_utils.de_normalize(preds, config['time_mean'], config['time_std'])
        return train_losses, test_losses, labels, preds

class FF_GRID(nn.Module):
    def __init__(self, model_name, n_features, hidden_size, embed_dict, device):
        super(FF_GRID, self).__init__()
        self.model_name = model_name
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.embed_dict = embed_dict
        self.device = device
        self.loss_fn = torch.nn.MSELoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        # self.driverID_em = nn.Embedding(embed_dict['driverID']['vocab_size'], embed_dict['driverID']['embed_dims'])
        # self.tripID_em = nn.Embedding(embed_dict['tripID']['vocab_size'], embed_dict['tripID']['embed_dims'])
        # Convolutional
        self.conv_stack = nn.Sequential(
            nn.Conv2d(4,1,3, padding=1),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Conv2d(1,1,3, padding=1),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Flatten()
        )
        # Feedforward
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_features + 4 + self.embed_total_dims, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_size, 1),
        )
    def forward(self, x):
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        # driverID_embedded = self.driverID_em(x_em[:,2])
        # tripID_embedded = self.tripID_em(x_em[:,3])
        # Convolute over whole grid
        x_gr = self.conv_stack(x_gr)
        # Feed data through the model
        # x = torch.cat([x_ct, x_gr, timeID_embedded, weekID_embedded, driverID_embedded, tripID_embedded], dim=1)
        x = torch.cat([x_ct, x_gr, timeID_embedded, weekID_embedded], dim=1)
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
    def fit_to_data(self, train_dataloader, test_dataloader, config, learn_rate, epochs):
        train_losses, test_losses = model_utils.train_model(self, train_dataloader, test_dataloader, learn_rate, epochs)
        labels, preds, avg_loss = model_utils.predict(self, test_dataloader)
        labels = data_utils.de_normalize(labels, config['time_mean'], config['time_std'])
        preds = data_utils.de_normalize(preds, config['time_mean'], config['time_std'])
        return train_losses, test_losses, labels, preds