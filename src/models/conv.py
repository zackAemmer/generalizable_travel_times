import numpy as np
import torch
from torch import nn

from models import masked_loss


class CONV(nn.Module):
    def __init__(self, model_name, input_size, output_size, hidden_size, batch_size, embed_dict):
        super(CONV, self).__init__()
        self.model_name = model_name
        self.loss_fn = masked_loss.MaskedMSELoss()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed_dict = embed_dict
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(embed_dict['timeID']['vocab_size'], embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(embed_dict['weekID']['vocab_size'], embed_dict['weekID']['embed_dims'])
        self.driverID_em = nn.Embedding(embed_dict['driverID']['vocab_size'], embed_dict['driverID']['embed_dims'])
        self.tripID_em = nn.Embedding(embed_dict['tripID']['vocab_size'], embed_dict['tripID']['embed_dims'])
        # Conv1d layer
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 3),
            nn.ReLu(),
            nn.Flatten()
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
        out, hidden_prev = self.conv(x_ct)
        # Add context, combine in linear layer
        embeddings = torch.cat((timeID_embedded,weekID_embedded,driverID_embedded,tripID_embedded), dim=1).unsqueeze(1)
        out = torch.cat((out, embeddings), dim=2)
        out = self.linear(out)
        out = out.squeeze(2)
        return out, hidden_prev