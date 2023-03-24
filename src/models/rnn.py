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
        self.loss_fn = masked_loss.MaskedMSELoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        self.driverID_em = nn.Embedding(embed_dict['driverID']['vocab_size'], embed_dict['driverID']['embed_dims'])
        self.tripID_em = nn.Embedding(embed_dict['tripID']['vocab_size'], embed_dict['tripID']['embed_dims'])
        # Recurrent layer
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        # Linear compression layer
        self.linear = nn.Linear(
            in_features=hidden_size + self.embed_total_dims,
            out_features=self.output_size
        )
    def forward(self, x, hidden_prev):
        x_em = x[0]
        x_ct = x[1]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        driverID_embedded = self.driverID_em(x_em[:,2])
        tripID_embedded = self.tripID_em(x_em[:,3])
        # Get recurrent pred
        out, hidden_prev = self.rnn(x_ct, hidden_prev)
        # x_ct_packed = torch.nn.utils.rnn.pack_padded_sequence(x_ct, x[2], batch_first=True)
        # out_packed, hidden_prev = self.rnn(x_ct_packed, hidden_prev)
        # out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        # Add context, combine in linear layer
        embeddings = torch.cat((timeID_embedded,weekID_embedded,driverID_embedded,tripID_embedded), dim=1).unsqueeze(1)
        embeddings = embeddings.repeat(1,out.shape[1],1)
        out = torch.cat((out, embeddings), dim=2)
        out = self.linear(out)
        out = out.squeeze(2)
        return out, hidden_prev
    def batch_step(self, data):
        inputs, labels = data
        inputs[:2] = [i.to(self.device) for i in inputs[:2]]
        labels = labels.to(self.device)
        hidden_prev = torch.zeros(1, len(data[1]), self.hidden_size).to(self.device)
        preds, hidden_prev = self(inputs, hidden_prev)
        hidden_prev = hidden_prev.detach()
        mask = data_utils.create_tensor_mask(inputs[2]).to(self.device)
        loss = self.loss_fn(preds, labels, mask)
        return labels, preds, loss
    def fit_to_data(self, train_dataloader, test_dataloader, test_mask, config, learn_rate, epochs):
        train_losses, test_losses = model_utils.train_model(self, train_dataloader, test_dataloader, learn_rate, epochs, sequential_flag=True)
        labels, preds, avg_loss = model_utils.predict(self, test_dataloader, sequential_flag=True)
        labels = data_utils.de_normalize(labels, config['time_calc_s_mean'], config['time_calc_s_std'])
        preds = data_utils.de_normalize(preds, config['time_calc_s_mean'], config['time_calc_s_std'])
        preds = data_utils.aggregate_tts(preds, test_mask)
        labels = data_utils.aggregate_tts(labels, test_mask)
        return train_losses, test_losses, labels, preds

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
        self.loss_fn = nn.MSELoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        self.driverID_em = nn.Embedding(embed_dict['driverID']['vocab_size'], embed_dict['driverID']['embed_dims'])
        self.tripID_em = nn.Embedding(embed_dict['tripID']['vocab_size'], embed_dict['tripID']['embed_dims'])
        # Recurrent layer
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        # Linear compression layer
        self.linear = nn.Linear(
            in_features=hidden_size + self.embed_total_dims,
            out_features=self.output_size
        )
    def forward(self, x, hidden_prev):
        x_em = x[0]
        x_ct = x[1]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        driverID_embedded = self.driverID_em(x_em[:,2])
        tripID_embedded = self.tripID_em(x_em[:,3])
        # Get recurrent pred
        out, hidden_prev = self.rnn(x_ct, hidden_prev)
        # Add context, combine in linear layer
        embeddings = torch.cat((timeID_embedded,weekID_embedded,driverID_embedded,tripID_embedded), dim=1)
        # Use only last element
        out = out[:,-1,:]
        out = torch.cat((out, embeddings), dim=1)
        out = self.linear(out).squeeze()
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
    def fit_to_data(self, train_dataloader, test_dataloader, config, learn_rate, epochs):
        train_losses, test_losses = model_utils.train_model(self, train_dataloader, test_dataloader, learn_rate, epochs)
        labels, preds, avg_loss = model_utils.predict(self, test_dataloader)
        labels = data_utils.de_normalize(labels, config['time_mean'], config['time_std'])
        preds = data_utils.de_normalize(preds, config['time_mean'], config['time_std'])
        return train_losses, test_losses, labels, preds