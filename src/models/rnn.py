import numpy as np
import torch
from torch import nn

from utils import data_utils, model_utils
from models import masked_loss, pos_encodings


class GRU(nn.Module):
    def __init__(self, model_name, n_features, hidden_size, batch_size, embed_dict, device):
        super(GRU, self).__init__()
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
        # Recurrent layer
        self.rnn = nn.GRU(input_size=self.n_features, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        # Linear compression layer
        self.linear = nn.Linear(in_features=self.hidden_size + self.embed_total_dims, out_features=1)
    def forward(self, x, hidden_prev):
        x_em = x[0]
        x_ct = x[1]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        # Get recurrent pred
        rnn_out, hidden_prev = self.rnn(x_ct, hidden_prev)
        rnn_out = self.activation(rnn_out)
        # Add context, combine in linear layer
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.repeat(1,rnn_out.shape[1],1)
        out = torch.cat((rnn_out, x_em), dim=2)
        out = self.linear(self.activation(out))
        out = out.squeeze(2)
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
        return labels, preds, loss
    def evaluate(self, test_dataloader, config):
        labels, preds, avg_batch_loss = model_utils.predict(self, test_dataloader, sequential_flag=True)
        labels = data_utils.de_normalize(labels, config['time_calc_s_mean'], config['time_calc_s_std'])
        preds = data_utils.de_normalize(preds, config['time_calc_s_mean'], config['time_calc_s_std'])
        _, mask = data_utils.get_seq_info(test_dataloader)
        preds = data_utils.aggregate_tts(preds, mask)
        labels = data_utils.aggregate_tts(labels, mask)
        return labels, preds

class GRU_GRID(nn.Module):
    def __init__(self, model_name, n_features, n_grid_features, hidden_size, grid_compression_size, batch_size, embed_dict, device):
        super(GRU_GRID, self).__init__()
        self.model_name = model_name
        self.n_features = n_features
        self.n_grid_features = n_grid_features
        self.hidden_size = hidden_size
        self.grid_compression_size = grid_compression_size
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
        # Grid Feedforward
        self.linear_relu_stack_grid = nn.Sequential(
            nn.Linear(self.n_grid_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.grid_compression_size),
            nn.ReLU()
        )
        # Recurrent layer
        self.rnn = nn.GRU(input_size=self.n_features + self.embed_total_dims + self.grid_compression_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        # Linear compression layer
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)
    def forward(self, x, hidden_prev):
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.repeat(1,x_ct.shape[1],1)
        # Feed grid data through model
        x_gr = torch.flatten(x_gr, 0, 1)
        x_gr = torch.flatten(x_gr, 1)
        x_gr = self.linear_relu_stack_grid(x_gr)
        x_gr = torch.reshape(x_gr, (x_ct.shape[0], x_ct.shape[1], x_gr.shape[1]))
        # Combine all variables
        x = torch.cat([x_em, x_gr, x_ct], dim=2)
        # Get recurrent pred
        rnn_out, hidden_prev = self.rnn(x, hidden_prev)
        rnn_out = self.activation(self.linear(self.activation(rnn_out)))
        rnn_out = rnn_out.squeeze(2)
        return rnn_out, hidden_prev
    def batch_step(self, data):
        inputs, labels = data
        inputs = [i.to(self.device) for i in inputs]
        labels = labels.to(self.device)
        hidden_prev = torch.zeros(1, len(data[1]), self.hidden_size).to(self.device)
        preds, hidden_prev = self(inputs, hidden_prev)
        hidden_prev = hidden_prev.detach()
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

class GRU_GRID_ATTN(nn.Module):
    def __init__(self, model_name, n_features, n_grid_features, n_channels, hidden_size, grid_compression_size, batch_size, embed_dict, device):
        super(GRU_GRID_ATTN, self).__init__()
        self.model_name = model_name
        self.n_features = n_features
        self.n_grid_features = n_grid_features
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.grid_compression_size = grid_compression_size
        self.batch_size = batch_size
        self.embed_dict = embed_dict
        self.device = device
        self.loss_fn = masked_loss.MaskedHuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        # 2d positional encoding
        self.pos_enc = pos_encodings.PositionalEncodingPermute2D(self.n_channels)
        # Activation layer
        self.activation = nn.ReLU()
        # Grid attention
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_grid_features, nhead=4, dim_feedforward=self.hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Grid Feedforward
        self.linear_relu_stack_grid = nn.Sequential(
            nn.Linear(self.n_grid_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.grid_compression_size),
            nn.ReLU()
        )
        # Recurrent layer
        self.rnn = nn.GRU(input_size=self.n_features + self.embed_total_dims + self.grid_compression_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        # Linear compression layer
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)
    def forward(self, x, hidden_prev):
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.repeat(1,x_ct.shape[1],1)
        # Feed grid data through model
        x_gr = torch.flatten(x_gr, 0, 1)
        x_gr = self.pos_enc(x_gr)
        x_gr = torch.flatten(x_gr, 1)
        x_gr = self.linear_relu_stack_grid(x_gr)
        x_gr = torch.reshape(x_gr, (x_ct.shape[0], x_ct.shape[1], x_gr.shape[1]))
        # Combine all variables
        x = torch.cat([x_em, x_gr, x_ct], dim=2)
        # Get recurrent pred
        rnn_out, hidden_prev = self.rnn(x, hidden_prev)
        rnn_out = self.activation(self.linear(self.activation(rnn_out)))
        rnn_out = rnn_out.squeeze(2)
        return rnn_out, hidden_prev
    def batch_step(self, data):
        inputs, labels = data
        inputs = [i.to(self.device) for i in inputs]
        labels = labels.to(self.device)
        hidden_prev = torch.zeros(1, len(data[1]), self.hidden_size).to(self.device)
        preds, hidden_prev = self(inputs, hidden_prev)
        hidden_prev = hidden_prev.detach()
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

# class GRU_RNN_MTO(nn.Module):
#     def __init__(self, model_name, input_size, hidden_size, batch_size, embed_dict, device):
#         super(GRU_RNN_MTO, self).__init__()
#         self.model_name = model_name
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.batch_size = batch_size
#         self.embed_dict = embed_dict
#         self.device = device
#         self.loss_fn = nn.HuberLoss()
#         # Embeddings
#         self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
#         self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
#         self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
#         # Activation layer
#         self.activation = nn.ReLU()
#         # Recurrent layer
#         self.rnn = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
#         # Linear compression layer
#         self.linear = nn.Linear(in_features=self.hidden_size + self.embed_total_dims, out_features=1)
#     def forward(self, x, hidden_prev):
#         x_em = x[0]
#         x_ct = x[1]
#         # Embed categorical variables
#         timeID_embedded = self.timeID_em(x_em[:,0])
#         weekID_embedded = self.weekID_em(x_em[:,1])
#         # Get recurrent pred
#         rnn_out, hidden_prev = self.rnn(x_ct, hidden_prev)
#         # Add context, combine in linear layer
#         embeddings = torch.cat((timeID_embedded,weekID_embedded), dim=1)
#         # Use only last element
#         rnn_out = rnn_out[:,-1,:]
#         out = torch.cat((rnn_out, embeddings), dim=1)
#         out = self.linear(self.activation(out))
#         out = out.squeeze()
#         return out, hidden_prev
#     def batch_step(self, data):
#         inputs, labels = data
#         inputs[:2] = [i.to(self.device) for i in inputs[:2]]
#         labels = labels.to(self.device)
#         hidden_prev = torch.zeros(1, len(data[1]), self.hidden_size).to(self.device)
#         preds, hidden_prev = self(inputs, hidden_prev)
#         hidden_prev = hidden_prev.detach()
#         loss = self.loss_fn(preds, labels)
#         return labels, preds, loss
#     def evaluate(self, test_dataloader, config):
#         labels, preds, avg_batch_loss = model_utils.predict(self, test_dataloader)
#         labels = data_utils.de_normalize(labels, config['time_mean'], config['time_std'])
#         preds = data_utils.de_normalize(preds, config['time_mean'], config['time_std'])
#         return labels, preds