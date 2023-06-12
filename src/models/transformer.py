import math

import numpy as np
import torch
from torch import nn

from utils import data_utils, model_utils
from models import masked_loss, pos_encodings


class TRSF(nn.Module):
    def __init__(self, model_name, n_features, hidden_size, batch_size, embed_dict, device):
        super(TRSF, self).__init__()
        self.model_name = model_name
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed_dict = embed_dict
        self.device = device
        self.train_time = 0.0
        self.loss_fn = masked_loss.MaskedHuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        # Activation layer
        self.activation = nn.ReLU()
        # Positional encoding layer
        self.pos_encoder = pos_encodings.PositionalEncoding1D(self.n_features)
        # Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_features, nhead=2, dim_feedforward=self.hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Linear compression layer
        self.feature_extract = nn.Linear(self.n_features + self.embed_total_dims, 1)
        self.feature_extract_activation = nn.ReLU()
    def forward(self, x):
        x_em = x[0]
        x_ct = x[1]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        # Get transformer prediction
        x_ct = self.pos_encoder(x_ct)
        x_ct = self.transformer_encoder(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        return out
    def batch_step(self, data):
        inputs, labels = data
        inputs = [i.to(self.device) for i in inputs]
        labels = labels.to(self.device)
        preds = self(inputs)
        mask = data_utils.create_tensor_mask(inputs[2]).to(self.device)
        loss = self.loss_fn(preds, labels, mask)
        return labels, preds, loss
    def evaluate(self, test_dataloader, config):
        labels, preds, avg_batch_loss = model_utils.predict(self, test_dataloader, sequential_flag=True)
        labels = data_utils.de_normalize(labels, config['time_calc_s_mean'], config['time_calc_s_std'])
        preds = data_utils.de_normalize(preds, config['time_calc_s_mean'], config['time_calc_s_std'])
        _, mask = data_utils.get_seq_info(test_dataloader)
        preds = data_utils.aggregate_tts(preds, mask)
        labels = data_utils.aggregate_tts(labels, mask)
        return labels, preds

class TRSF_GRID(nn.Module):
    def __init__(self, model_name, n_features, n_grid_features, hidden_size, grid_compression_size, batch_size, embed_dict, device):
        super(TRSF_GRID, self).__init__()
        self.model_name = model_name
        self.n_features = n_features
        self.n_grid_features = n_grid_features
        self.hidden_size = hidden_size
        self.grid_compression_size = grid_compression_size
        self.batch_size = batch_size
        self.embed_dict = embed_dict
        self.device = device
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
        # Positional encoding layer
        self.pos_encoder = pos_encodings.PositionalEncoding1D(self.n_features + self.grid_compression_size)
        # Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_features + self.grid_compression_size, nhead=2, dim_feedforward=self.hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Linear compression layer
        self.feature_extract = nn.Linear(self.n_features + self.embed_total_dims + self.grid_compression_size, 1)
        self.feature_extract_activation = nn.ReLU()
    def forward(self, x):
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
        # Get transformer prediction
        x_ct = torch.cat((x_ct, x_gr), dim=2)
        x_ct = self.pos_encoder(x_ct)
        x_ct = self.transformer_encoder(x_ct)
        # Combine all variables
        out = torch.cat((x_em, x_ct), dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        return out
    def batch_step(self, data):
        inputs, labels = data
        inputs = [i.to(self.device) for i in inputs]
        labels = labels.to(self.device)
        preds = self(inputs)
        mask = data_utils.create_tensor_mask(inputs[-1]).to(self.device)
        loss = self.loss_fn(preds, labels, mask)
        return labels, preds, loss
    def evaluate(self, test_dataloader, config):
        labels, preds, avg_batch_loss = model_utils.predict(self, test_dataloader, sequential_flag=True)
        labels = data_utils.de_normalize(labels, config['time_calc_s_mean'], config['time_calc_s_std'])
        preds = data_utils.de_normalize(preds, config['time_calc_s_mean'], config['time_calc_s_std'])
        _, mask = data_utils.get_seq_info(test_dataloader)
        preds = data_utils.aggregate_tts(preds, mask)
        labels = data_utils.aggregate_tts(labels, mask)
        return labels, preds

class TRSF_GRID_ATTN(nn.Module):
    def __init__(self, model_name, n_features, n_grid_features, n_channels, hidden_size, grid_compression_size, batch_size, embed_dict, device):
        super(TRSF_GRID_ATTN, self).__init__()
        self.model_name = model_name
        self.n_features = n_features
        self.n_grid_features = n_grid_features
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.grid_compression_size = grid_compression_size
        self.batch_size = batch_size
        self.embed_dict = embed_dict
        self.device = device
        self.train_time = 0.0
        self.loss_fn = masked_loss.MaskedHuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        # Grid
        self.grid_pos_encoder = pos_encodings.PositionalEncoding3D(self.n_channels)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.n_features, num_heads=2, dropout=0.1, batch_first=True, kdim=self.n_grid_features, vdim=self.n_grid_features)
        self.activation = nn.ReLU()
        self.trsf_ff = nn.Linear(in_features=self.n_features, out_features=self.hidden_size)
        # Encoder layer
        self.seq_pos_encoder = pos_encodings.PositionalEncoding1D(self.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=2, dim_feedforward=self.hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Linear compression layer
        self.feature_extract = nn.Linear(self.hidden_size + self.embed_total_dims, 1)
        self.feature_extract_activation = nn.ReLU()
    def forward(self, x):
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        # Cross attention between continuous inputs and grid features in a given trajectory
        x_gr = torch.flatten(x_gr, 2, 3)
        x_gr = torch.swapaxes(x_gr, 2, 4)
        x_gr = self.grid_pos_encoder(x_gr)
        x_gr = torch.flatten(x_gr, 2)
        out, attn_wts = self.cross_attn(query=x_ct, key=x_gr, value=x_gr)
        out = self.trsf_ff(self.activation(out))
        # Get sequence prediction
        out = self.seq_pos_encoder(out)
        out = self.transformer_encoder(out)
        # Combine all variables
        out = torch.cat((x_em, out), dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        return out
    def batch_step(self, data):
        inputs, labels = data
        inputs = [i.to(self.device) for i in inputs]
        labels = labels.to(self.device)
        preds = self(inputs)
        mask = data_utils.create_tensor_mask(inputs[-1]).to(self.device)
        loss = self.loss_fn(preds, labels, mask)
        return labels, preds, loss
    def evaluate(self, test_dataloader, config):
        labels, preds, avg_batch_loss = model_utils.predict(self, test_dataloader, sequential_flag=True)
        labels = data_utils.de_normalize(labels, config['time_calc_s_mean'], config['time_calc_s_std'])
        preds = data_utils.de_normalize(preds, config['time_calc_s_mean'], config['time_calc_s_std'])
        _, mask = data_utils.get_seq_info(test_dataloader)
        preds = data_utils.aggregate_tts(preds, mask)
        labels = data_utils.aggregate_tts(labels, mask)
        return labels, preds