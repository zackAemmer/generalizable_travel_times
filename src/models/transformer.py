import math

import numpy as np
import torch
from torch import nn

from utils import data_utils, model_utils
from models import masked_loss


class TRANSFORMER(nn.Module):
    def __init__(self, model_name, input_size, output_size, hidden_size, batch_size, embed_dict, device):
        super(TRANSFORMER, self).__init__()
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
        # Activation layer
        self.activation = nn.ReLU()
        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(self.input_size+self.embed_total_dims)
        # Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size+self.embed_total_dims, nhead=2, dim_feedforward=self.hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Linear compression layer
        self.linear = nn.Linear(self.input_size + self.embed_total_dims, self.output_size)
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
        out = self.activation(self.linear(out)).squeeze(2)
        return out
    def batch_step(self, data):
        inputs, labels = data
        inputs = [i.to(self.device) for i in inputs]
        labels = labels.to(self.device)
        preds = self(inputs)
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        z = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)