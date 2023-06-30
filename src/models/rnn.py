import numpy as np
import torch
from torch import nn

from utils import data_utils, model_utils
from models import masked_loss, pos_encodings


class GRU(nn.Module):
    def __init__(self, model_name, n_features, hidden_size, batch_size, collate_fn, embed_dict, device):
        super(GRU, self).__init__()
        self.model_name = model_name
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.embed_dict = embed_dict
        self.device = device
        self.is_nn = True
        self.requires_grid = False
        self.train_time = 0.0
        self.loss_fn = masked_loss.MaskedHuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        # Activation layer
        self.activation = nn.ReLU()
        # Recurrent layer
        self.rnn = nn.GRU(input_size=self.n_features, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        # Linear compression layer
        self.feature_extract = nn.Linear(in_features=self.hidden_size + self.embed_total_dims, out_features=1)
        self.feature_extract_activation = nn.ReLU()
    def forward(self, x, hidden_prev):
        x_em = x[0]
        x_ct = x[1]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        # Get recurrent pred
        x_ct, hidden_prev = self.rnn(x_ct, hidden_prev)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        return out, hidden_prev
    def batch_step(self, data):
        inputs, labels = data
        inputs = [i.to(self.device) for i in inputs]
        labels = labels.to(self.device)
        hidden_prev = torch.zeros(1, len(data[1]), self.hidden_size).to(self.device)
        preds, hidden_prev = self(inputs, hidden_prev)
        hidden_prev = hidden_prev.detach()
        mask = data_utils.create_tensor_mask(inputs[-1]).to(self.device)
        loss = self.loss_fn(preds, labels, mask)
        return labels, preds, loss, inputs[-1]
    def evaluate(self, test_dataloader, config):
        labels, preds, avg_batch_loss, seq_lens = model_utils.predict(self, test_dataloader, sequential_flag=True)
        labels = data_utils.de_normalize(labels, config['time_calc_s_mean'], config['time_calc_s_std'])
        preds = data_utils.de_normalize(preds, config['time_calc_s_mean'], config['time_calc_s_std'])
        mask = data_utils.create_tensor_mask(torch.cat(seq_lens)).numpy()
        preds = data_utils.aggregate_tts(preds, mask)
        labels = data_utils.aggregate_tts(labels, mask)
        return labels, preds

class GRU_GRID(nn.Module):
    def __init__(self, model_name, n_features, n_grid_features, hidden_size, grid_compression_size, batch_size, collate_fn, embed_dict, device):
        super(GRU_GRID, self).__init__()
        self.model_name = model_name
        self.n_features = n_features
        self.n_grid_features = n_grid_features
        self.hidden_size = hidden_size
        self.grid_compression_size = grid_compression_size
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.embed_dict = embed_dict
        self.device = device
        self.is_nn = True
        self.requires_grid = True
        self.train_time = 0.0
        self.loss_fn = masked_loss.MaskedHuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        # Activation layer
        self.activation = nn.ReLU()
        # Grid Feedforward
        self.linear_relu_stack_grid = nn.Sequential(
            nn.Linear(self.n_grid_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.grid_compression_size),
            nn.ReLU()
        )
        # Recurrent layer
        self.rnn = nn.GRU(input_size=self.n_features + self.grid_compression_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        # Linear compression layer
        self.feature_extract = nn.Linear(in_features=self.hidden_size + self.embed_total_dims, out_features=1)
        self.feature_extract_activation = nn.ReLU()
    def forward(self, x, hidden_prev):
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        # Feed grid data through model
        x_gr = torch.flatten(x_gr, 2)
        x_gr = self.linear_relu_stack_grid(x_gr)
        # Get recurrent pred
        x_ct = torch.cat([x_ct, x_gr], dim=2)
        x_ct, hidden_prev = self.rnn(x_ct, hidden_prev)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        return out, hidden_prev
    def batch_step(self, data):
        inputs, labels = data
        inputs = [i.to(self.device) for i in inputs]
        labels = labels.to(self.device)
        hidden_prev = torch.zeros(1, len(data[1]), self.hidden_size).to(self.device)
        preds, hidden_prev = self(inputs, hidden_prev)
        hidden_prev = hidden_prev.detach()
        mask = data_utils.create_tensor_mask(inputs[-1]).to(self.device)
        loss = self.loss_fn(preds, labels, mask)
        return labels, preds, loss, inputs[-1]
    def evaluate(self, test_dataloader, config):
        labels, preds, avg_batch_loss, seq_lens = model_utils.predict(self, test_dataloader, sequential_flag=True)
        labels = data_utils.de_normalize(labels, config['time_calc_s_mean'], config['time_calc_s_std'])
        preds = data_utils.de_normalize(preds, config['time_calc_s_mean'], config['time_calc_s_std'])
        mask = data_utils.create_tensor_mask(torch.cat(seq_lens)).numpy()
        preds = data_utils.aggregate_tts(preds, mask)
        labels = data_utils.aggregate_tts(labels, mask)
        return labels, preds