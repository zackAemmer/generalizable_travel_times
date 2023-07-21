import numpy as np
import torch
from torch import nn
import lightning.pytorch as pl

from utils import data_utils, model_utils
from models import masked_loss, pos_encodings


class GRU_L(pl.LightningModule):
    def __init__(self, model_name, n_features, hyperparameter_dict, embed_dict, collate_fn, config):
        super(GRU_L, self).__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.n_features = n_features
        self.hyperparameter_dict = hyperparameter_dict
        self.embed_dict = embed_dict
        self.collate_fn = collate_fn
        self.config = config
        self.is_nn = True
        self.requires_grid = False
        self.train_time = 0.0
        self.loss_fn = masked_loss.MaskedHuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(self.embed_dict['timeID']['vocab_size'], self.embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(self.embed_dict['weekID']['vocab_size'], self.embed_dict['weekID']['embed_dims'])
        # Recurrent layer
        self.norm = nn.BatchNorm1d(self.n_features)
        self.rnn = nn.GRU(input_size=self.n_features, hidden_size=self.hyperparameter_dict['HIDDEN_SIZE'], num_layers=self.hyperparameter_dict['NUM_LAYERS'], batch_first=True, dropout=self.hyperparameter_dict['DROPOUT_RATE'])
        # Linear compression layer
        self.feature_extract = nn.Linear(in_features=self.hyperparameter_dict['HIDDEN_SIZE'] + self.embed_total_dims, out_features=1)
        self.feature_extract_activation = nn.ReLU()
    def training_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_sl = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        # Get recurrent pred
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.norm(x_ct)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct, hidden_prev = self.rnn(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl).to(x_sl)
        loss = self.loss_fn(out, y, mask)
        self.log(f"{self.model_name}_train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_sl = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        # Get recurrent pred
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.norm(x_ct)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct, hidden_prev = self.rnn(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl).to(x_sl)
        loss = self.loss_fn(out, y, mask)
        self.log(f"{self.model_name}_valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def test_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_sl = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        # Get recurrent pred
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.norm(x_ct)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct, hidden_prev = self.rnn(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl).to(x_sl)
        loss = self.loss_fn(out, y, mask)
        self.log(f"{self.model_name}_test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def predict_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_sl = x[2]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        # Get recurrent pred
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.norm(x_ct)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct, hidden_prev = self.rnn(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl).to(x_sl)
        out = out.cpu().numpy()
        mask = mask.cpu().numpy()
        y = y.cpu().numpy()
        out = data_utils.aggregate_tts(out, mask)
        y = data_utils.aggregate_tts(y, mask)
        return (out, y)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class GRU_GRID_L(pl.LightningModule):
    def __init__(self, model_name, n_features, n_grid_features, grid_compression_size, hyperparameter_dict, embed_dict, collate_fn, config):
        super(GRU_GRID_L, self).__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.n_features = n_features
        self.n_grid_features = n_grid_features
        self.grid_compression_size = grid_compression_size
        self.hyperparameter_dict = hyperparameter_dict
        self.embed_dict = embed_dict
        self.collate_fn = collate_fn
        self.config = config
        self.is_nn = True
        self.requires_grid = True
        self.train_time = 0.0
        self.loss_fn = masked_loss.MaskedHuberLoss()
        # Embeddings
        self.embed_total_dims = np.sum([self.embed_dict[key]['embed_dims'] for key in self.embed_dict.keys()]).astype('int32')
        self.timeID_em = nn.Embedding(self.embed_dict['timeID']['vocab_size'], self.embed_dict['timeID']['embed_dims'])
        self.weekID_em = nn.Embedding(self.embed_dict['weekID']['vocab_size'], self.embed_dict['weekID']['embed_dims'])
        # Grid Feedforward
        self.grid_norm = nn.BatchNorm1d(self.n_grid_features)
        self.linear_relu_stack_grid = nn.Sequential(
            nn.Linear(self.n_grid_features, self.hyperparameter_dict['HIDDEN_SIZE']),
            nn.ReLU(),
            nn.Linear(self.hyperparameter_dict['HIDDEN_SIZE'], self.grid_compression_size),
            nn.ReLU()
        )
        # Recurrent layer
        self.norm = nn.BatchNorm1d(self.n_features + self.grid_compression_size)
        self.rnn = nn.GRU(input_size=self.n_features + self.grid_compression_size, hidden_size=self.hyperparameter_dict['HIDDEN_SIZE'], num_layers=self.hyperparameter_dict['NUM_LAYERS'], batch_first=True, dropout=self.hyperparameter_dict['DROPOUT_RATE'])
        # Linear compression layer
        self.feature_extract = nn.Linear(in_features=self.hyperparameter_dict['HIDDEN_SIZE'] + self.embed_total_dims, out_features=1)
        self.feature_extract_activation = nn.ReLU()
    def training_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        x_sl = x[3]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        # Feed grid data through model
        x_gr = torch.flatten(x_gr, 2)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.grid_norm(x_gr)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.linear_relu_stack_grid(x_gr)
        # Get recurrent pred
        x_ct = torch.cat([x_ct, x_gr], dim=2)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.norm(x_ct)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct, hidden_prev = self.rnn(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl).to(x_sl)
        loss = self.loss_fn(out, y, mask)
        self.log(f"{self.model_name}_train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        x_sl = x[3]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        # Feed grid data through model
        x_gr = torch.flatten(x_gr, 2)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.grid_norm(x_gr)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.linear_relu_stack_grid(x_gr)
        # Get recurrent pred
        x_ct = torch.cat([x_ct, x_gr], dim=2)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.norm(x_ct)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct, hidden_prev = self.rnn(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl).to(x_sl)
        loss = self.loss_fn(out, y, mask)
        self.log(f"{self.model_name}_valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def test_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        x_sl = x[3]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        # Feed grid data through model
        x_gr = torch.flatten(x_gr, 2)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.grid_norm(x_gr)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.linear_relu_stack_grid(x_gr)
        # Get recurrent pred
        x_ct = torch.cat([x_ct, x_gr], dim=2)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.norm(x_ct)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct, hidden_prev = self.rnn(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl).to(x_sl)
        loss = self.loss_fn(out, y, mask)
        self.log(f"{self.model_name}_test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def predict_step(self, batch, batch_idx):
        x,y = batch
        x_em = x[0]
        x_ct = x[1]
        x_gr = x[2]
        x_sl = x[3]
        # Embed categorical variables
        timeID_embedded = self.timeID_em(x_em[:,0])
        weekID_embedded = self.weekID_em(x_em[:,1])
        x_em = torch.cat((timeID_embedded,weekID_embedded), dim=1).unsqueeze(1)
        x_em = x_em.expand(-1, x_ct.shape[1], -1)
        # Feed grid data through model
        x_gr = torch.flatten(x_gr, 2)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.grid_norm(x_gr)
        x_gr = torch.swapaxes(x_gr, 1, 2)
        x_gr = self.linear_relu_stack_grid(x_gr)
        # Get recurrent pred
        x_ct = torch.cat([x_ct, x_gr], dim=2)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct = self.norm(x_ct)
        x_ct = torch.swapaxes(x_ct, 1, 2)
        x_ct, hidden_prev = self.rnn(x_ct)
        # Combine all variables
        out = torch.cat([x_em, x_ct], dim=2)
        out = self.feature_extract(self.feature_extract_activation(out)).squeeze(2)
        mask = data_utils.create_tensor_mask(x_sl).to(x_sl)
        out = out.cpu().numpy()
        mask = mask.cpu().numpy()
        y = y.cpu().numpy()
        out = data_utils.aggregate_tts(out, mask)
        y = data_utils.aggregate_tts(y, mask)
        return (out, y)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer