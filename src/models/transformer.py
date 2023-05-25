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
        self.loss_fn = masked_loss.MaskedHuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        # Activation layer
        self.activation = nn.ReLU()
        # Positional encoding layer
        self.pos_encoder = pos_encodings.PositionalEncoding(self.n_features+self.embed_total_dims)
        # Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_features+self.embed_total_dims, nhead=4, dim_feedforward=self.hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Linear compression layer
        self.linear = nn.Linear(self.n_features + self.embed_total_dims, 1)
    def forward(self, x):
        x_em, x_ct, slens = x
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded, weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.repeat(1,x_ct.shape[1],1)
        x = torch.cat((x_ct, x_em), dim=2)
        # Get transformer prediction
        out = self.pos_encoder(x)
        out = self.transformer_encoder(out)
        out = self.linear(self.activation(out)).squeeze(2)
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

# class TRSF_GRID(nn.Module):
#     def __init__(self, model_name, n_features, hidden_size, batch_size, embed_dict, device):
#         super(TRANSFORMER_GRID, self).__init__()
#         self.model_name = model_name
#         self.n_features = n_features
#         self.hidden_size = hidden_size
#         self.batch_size = batch_size
#         self.embed_dict = embed_dict
#         self.device = device
#         self.loss_fn = masked_loss.MaskedMSELoss()

#         # Embeddings
#         self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
#         self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
#         self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
#         # Activation layer
#         self.activation = nn.ReLU()
#         # Linear compression layer
#         self.linear = nn.Linear(self.n_features + self.embed_total_dims, 1)
#         # Flat layer
#         self.flatten = nn.Flatten(start_dim=1)

#         # 3d positional encoding for grid/positions
#         self.pos_encoder_grid = PositionalEncodingPermute3D(4)
#         # Cross attention for grid
#         self.cross_attn = nn.MultiheadAttention(embed_dim=self.n_features+self.embed_total_dims, num_heads=4, dropout=0.1, batch_first=True, kdim=1, vdim=1)

#         # 1d positional encoding layer for sequence
#         self.pos_encoder = pos_encodings.PositionalEncoding(self.n_features+self.embed_total_dims)
#         # Self attention for sequence
#         encoder_seq = nn.TransformerEncoderLayer(d_model=self.n_features+self.embed_total_dims, nhead=4, dropout=0.1, dim_feedforward=self.hidden_size, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_seq, num_layers=2)

#     def forward(self, x):
#         x_em, x_ct, slens, x_gr = x
#         # Embed categorical variables
#         timeID_embedded = self.timeID_em(x_em[:,0])
#         weekID_embedded = self.weekID_em(x_em[:,1])
#         x_em = torch.cat((timeID_embedded, weekID_embedded), dim=1).unsqueeze(1)
#         # Join continuous and categorical variables, as queries for cross attention
#         x_em = x_em.repeat(1,x_ct.shape[1],1)
#         x_query = torch.cat((x_ct, x_em), dim=2)
#         # Add 3d positional encoding to grid features along channel dimension
#         x_key = torch.swapaxes(x_gr, 1, 2)
#         penc = self.pos_encoder_grid(x_key)
#         x_key = x_key + penc
#         x_key = x_key[:,:,0,:,:]
#         x_key = self.flatten(x_key)
#         x_key = x_key.unsqueeze(2)
#         # Get cross attention on grid
#         cross_attn_out, cross_attn_wts = self.cross_attn(query=x_query, key=x_key, value=x_key)
#         # Get self-attention for sequence
#         out = self.pos_encoder(cross_attn_out)
#         out = self.transformer(out)
#         out = self.linear(self.activation(out)).squeeze(2)
#         return out
#     def batch_step(self, data):
#         inputs, labels = data
#         inputs = [i.to(self.device) for i in inputs]
#         labels = labels.to(self.device)
#         preds = self(inputs)
#         mask = data_utils.create_tensor_mask(inputs[2]).to(self.device)
#         loss = self.loss_fn(preds, labels, mask)
#         return labels, preds, loss
#     def fit_to_data(self, train_dataloader, test_dataloader, test_mask, config, learn_rate, epochs):
#         train_losses, test_losses = model_utils.train_model(self, train_dataloader, test_dataloader, learn_rate, epochs, sequential_flag=True)
#         labels, preds, avg_loss = model_utils.predict(self, test_dataloader, sequential_flag=True)
#         labels = data_utils.de_normalize(labels, config['time_calc_s_mean'], config['time_calc_s_std'])
#         preds = data_utils.de_normalize(preds, config['time_calc_s_mean'], config['time_calc_s_std'])
#         preds = data_utils.aggregate_tts(preds, test_mask)
#         labels = data_utils.aggregate_tts(labels, test_mask)
#         return train_losses, test_losses, labels, preds