import numpy as np
import torch
from torch import nn

from utils import data_utils, model_utils
from models import masked_loss


class GRU_RNN(nn.Module):
    def __init__(self, model_name, input_size, output_size, hidden_size, batch_size, embed_dict, device):
        super(GRU_RNN, self).__init__()
        self.model_name = model_name
        self.input_size = input_size
        self.output_size = output_size
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
        self.rnn = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        # Linear compression layer
        self.linear = nn.Linear(in_features=self.hidden_size + self.embed_total_dims, out_features=self.output_size)
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
        embeddings = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        embeddings = embeddings.repeat(1,rnn_out.shape[1],1)
        out = torch.cat((rnn_out, embeddings), dim=2)
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

class GRU_RNN_MTO(nn.Module):
    def __init__(self, model_name, input_size, output_size, hidden_size, batch_size, embed_dict, device):
        super(GRU_RNN_MTO, self).__init__()
        self.model_name = model_name
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed_dict = embed_dict
        self.device = device
        self.loss_fn = nn.HuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        # Activation layer
        self.activation = nn.ReLU()
        # Recurrent layer
        self.rnn = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        # Linear compression layer
        self.linear = nn.Linear(in_features=self.hidden_size + self.embed_total_dims, out_features=self.output_size)
    def forward(self, x, hidden_prev):
        x_em = x[0]
        x_ct = x[1]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        # Get recurrent pred
        rnn_out, hidden_prev = self.rnn(x_ct, hidden_prev)
        # Add context, combine in linear layer
        embeddings = torch.cat((timeID_embedded,weekID_embedded), dim=1)
        # Use only last element
        rnn_out = rnn_out[:,-1,:]
        out = torch.cat((rnn_out, embeddings), dim=1)
        out = self.linear(self.activation(out))
        out = out.squeeze()
        return out, hidden_prev
    def batch_step(self, data):
        inputs, labels = data
        inputs[:2] = [i.to(self.device) for i in inputs[:2]]
        labels = labels.to(self.device)
        hidden_prev = torch.zeros(1, len(data[1]), self.hidden_size).to(self.device)
        preds, hidden_prev = self(inputs, hidden_prev)
        hidden_prev = hidden_prev.detach()
        loss = self.loss_fn(preds, labels)
        return labels, preds, loss
    def evaluate(self, test_dataloader, config):
        labels, preds, avg_batch_loss = model_utils.predict(self, test_dataloader)
        labels = data_utils.de_normalize(labels, config['time_mean'], config['time_std'])
        preds = data_utils.de_normalize(preds, config['time_mean'], config['time_std'])
        return labels, preds

# class GRU_RNN_GRID(nn.Module):
#     def __init__(self, model_name, input_size, output_size, hidden_size, batch_size, embed_dict, device):
#         super(GRU_RNN_GRID, self).__init__()
#         self.model_name = model_name
#         self.input_size = input_size
#         self.output_size = output_size
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
#         # Recurrent layer
#         self.rnn = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
#         # Linear compression layer
#         self.linear = nn.Linear(in_features=self.hidden_size + self.embed_total_dims, out_features=self.output_size)
#     def forward(self, x, hidden_prev):
#         x_em = x[0]
#         x_ct = x[1]
#         # Embed categorical variables
#         timeID_embedded = self.timeID_em(x_em[:,0])
#         weekID_embedded = self.weekID_em(x_em[:,1])
#         # Get recurrent pred
#         rnn_out, hidden_prev = self.rnn(x_ct, hidden_prev)
#         rnn_out = self.activation(rnn_out)
#         # Add context, combine in linear layer
#         # embeddings = torch.cat((timeID_embedded,weekID_embedded,driverID_embedded,tripID_embedded), dim=1).unsqueeze(1)
#         embeddings = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
#         embeddings = embeddings.repeat(1,rnn_out.shape[1],1)
#         out = torch.cat((rnn_out, embeddings), dim=2)
#         out = self.linear(self.activation(out))
#         out = out.squeeze(2)
#         return out, hidden_prev
#     def batch_step(self, data):
#         inputs, labels = data
#         inputs = [i.to(self.device) for i in inputs]
#         labels = labels.to(self.device)
#         hidden_prev = torch.zeros(1, len(data[1]), self.hidden_size).to(self.device)
#         preds, hidden_prev = self(inputs, hidden_prev)
#         hidden_prev = hidden_prev.detach()
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

# class GRU_RNN_GRID_CONV(nn.Module):
#     def __init__(self, model_name, input_size, output_size, grid_pt_size, hidden_size, batch_size, embed_dict, device):
#         super(GRU_RNN_GRID_CONV, self).__init__()
#         self.model_name = model_name
#         self.input_size = input_size
#         self.output_size = output_size
#         self.grid_pt_size = grid_pt_size
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
#         # Recurrent layer
#         self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
#         # Convolution layer
#         self.conv = nn.Conv2d(in_channels=4, out_channels=self.hidden_size, kernel_size=3, padding=1)
#         # Pooling layer
#         self.pool = nn.AvgPool2d(kernel_size=4)
#         self.flatten = nn.Flatten(start_dim=1)
#         # Linear compression layer
#         self.linear = nn.Linear(in_features=self.hidden_size + 128 + self.embed_total_dims, out_features=self.output_size)
#     def forward(self, x, hidden_prev):
#         x_em = x[0]
#         x_ct = x[1]
#         x_gr = x[3]
#         # Embed categorical variables
#         timeID_embedded = self.timeID_em(x_em[:,0])
#         weekID_embedded = self.weekID_em(x_em[:,1])
#         # Get recurrent pred
#         rnn_out, hidden_prev = self.rnn(x_ct, hidden_prev)
#         rnn_out = self.activation(rnn_out)
#         # Get conv pred
#         conv_out = torch.reshape(x_gr, (x_gr.shape[0]*x_gr.shape[1], x_gr.shape[2], x_gr.shape[3], x_gr.shape[4]))
#         conv_out = self.conv(conv_out)
#         conv_out = self.activation(self.pool(conv_out))
#         conv_out = self.flatten(conv_out)
#         conv_out = torch.reshape(conv_out, (x_gr.shape[0], x_gr.shape[1], conv_out.shape[-1]))
#         # Add context, combine in linear layer
#         embeddings = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
#         embeddings = embeddings.repeat(1,rnn_out.shape[1],1)
#         out = torch.cat((rnn_out, conv_out, embeddings), dim=2)
#         out = self.linear(self.activation(out))
#         out = out.squeeze(2)
#         return out, hidden_prev
#     def batch_step(self, data):
#         inputs, labels = data
#         inputs = [i.to(self.device) for i in inputs]
#         labels = labels.to(self.device)
#         hidden_prev = torch.zeros(1, len(data[1]), self.hidden_size).to(self.device)
#         preds, hidden_prev = self(inputs, hidden_prev)
#         hidden_prev = hidden_prev.detach()
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

# class LSTM_RNN(nn.Module):
#     def __init__(self, model_name, input_size, output_size, hidden_size, batch_size, embed_dict, device):
#         super(LSTM_RNN, self).__init__()
#         self.model_name = model_name
#         self.input_size = input_size
#         self.output_size = output_size
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
#         # Recurrent layer
#         self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
#         # Linear compression layer
#         self.linear = nn.Linear(in_features=self.hidden_size + self.embed_total_dims, out_features=self.output_size)
#     def forward(self, x, initial_state):
#         x_em = x[0]
#         x_ct = x[1]
#         h_0, c_0 = initial_state
#         # Embed categorical variables
#         timeID_embedded = self.timeID_em(x_em[:,0])
#         weekID_embedded = self.weekID_em(x_em[:,1])
#         # Get recurrent pred
#         rnn_out, prev_state = self.rnn(x_ct, (h_0, c_0))
#         rnn_out = self.activation(rnn_out)
#         # Add context, combine in linear layer
#         embeddings = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
#         embeddings = embeddings.repeat(1,rnn_out.shape[1],1)
#         out = torch.cat((rnn_out, embeddings), dim=2)
#         out = self.linear(self.activation(out))
#         out = out.squeeze(2)
#         return out, prev_state
#     def batch_step(self, data):
#         inputs, labels = data
#         inputs = [i.to(self.device) for i in inputs]
#         labels = labels.to(self.device)
#         h_0 = torch.zeros(1, len(data[1]), self.hidden_size).to(self.device)
#         c_0 = torch.zeros(1, len(data[1]), self.hidden_size).to(self.device)
#         preds, (h_prev, c_prev) = self(inputs, (h_0, c_0))
#         h_prev = h_prev.detach()
#         c_prev = c_prev.detach()
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