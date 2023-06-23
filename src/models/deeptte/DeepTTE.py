import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import data_utils, deeptte_utils, model_utils
from models.deeptte import Attr, SpatioTemporal
import numpy as np


EPS = 10

class EntireEstimator(nn.Module):
    def __init__(self, input_size, num_final_fcs, hidden_size = 128):
        super(EntireEstimator, self).__init__()

        self.input2hid = nn.Linear(input_size, hidden_size)

        ### num_final_fcs refers to number of fully connected layers with residual connections 
        self.residuals = nn.ModuleList()
        for i in range(num_final_fcs):
            self.residuals.append(nn.Linear(hidden_size, hidden_size))

        ### Final fully connected layer to output the Collective TTE value
        self.hid2out = nn.Linear(hidden_size, 1)

    def forward(self, attr_t, sptm_t):
        inputs = torch.cat((attr_t, sptm_t), dim = 1)   ### size [batch, 128 + 28 = 156]

        hidden = F.leaky_relu(self.input2hid(inputs))   
        ### Non-linear mapping of concatenated vector used as residual connection for first residual FC layer
        ### size [batch, hidden_size = 128]

        for i in range(len(self.residuals)):
            residual = F.leaky_relu(self.residuals[i](hidden))
            hidden = hidden + residual

        out = self.hid2out(hidden)   ### size [batch, 1]: final Collective TTE value

        return out

    def eval_on_batch(self, pred, label, mean, std):
        label = label.view(-1, 1)   ### size [batch, 1]

        ### Un-normalising the labels and predictions
        label = label * std + mean
        pred = pred * std + mean

        ### MAPE loss
        loss = torch.abs(pred - label) / label   ### size [batch, 1]

        return {'label': label, 'pred': pred}, loss.mean()


class LocalEstimator(nn.Module):
    def __init__(self, input_size):
        super(LocalEstimator, self).__init__()

        self.input2hid = nn.Linear(input_size, 64)
        self.hid2hid = nn.Linear(64, 32)
        self.hid2out = nn.Linear(32, 1)

    def forward(self, sptm_s):
        hidden = F.leaky_relu(self.input2hid(sptm_s))   ### size [variable, 64]

        hidden = F.leaky_relu(self.hid2hid(hidden))   ### size [variable, 32]

        out = self.hid2out(hidden)   ### size [variable, 1]

        return out

    def eval_on_batch(self, pred, lens, label, mean, std):
        label = nn.utils.rnn.pack_padded_sequence(label, lens, batch_first = True, enforce_sorted=False)[0]
        ### Since predictions are packed, labels have to be packed too
        label = label.view(-1, 1)

        label = label * std + mean
        pred = pred * std + mean

        ### MAPE loss
        loss = torch.abs(pred - label) / (label + EPS)   ### size [variable, 1]
        
        return loss.mean()


class Net(nn.Module):
    def __init__(self, model_name, collate_fn, device, config, kernel_size = 3, num_filter = 32, pooling_method = 'attention', num_final_fcs = 3, final_fc_size = 128, alpha = 0.3, cfg=None):
        super(Net, self).__init__()

        # Training configurations
        self.model_name = model_name
        self.collate_fn = collate_fn
        self.device = device
        self.requires_grid = False
        self.config = config
        self.train_time = 0.0

        # parameter of attribute / spatio-temporal component
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method

        # parameter of multi-task learning component
        self.num_final_fcs = num_final_fcs
        self.final_fc_size = final_fc_size
        self.alpha = alpha

        # Number of embeddings for vehID
        self.cfg = cfg

        self.build()
        self.init_weight()

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.xavier_uniform_(param.data)

    def build(self):
        # attribute component
        self.attr_net = Attr.Net(cfg=self.cfg)

        # spatio-temporal component
        self.spatio_temporal = SpatioTemporal.Net(attr_size = self.attr_net.out_size(), \
                                                       kernel_size = self.kernel_size, \
                                                       num_filter = self.num_filter, \
                                                       pooling_method = self.pooling_method
        )

        self.entire_estimate = EntireEstimator(input_size =  self.spatio_temporal.out_size() + self.attr_net.out_size(), num_final_fcs = self.num_final_fcs, hidden_size = self.final_fc_size)

        self.local_estimate = LocalEstimator(input_size = self.spatio_temporal.out_size())


    def forward(self, attr, traj, config):
        attr_t = self.attr_net(attr)   ### shape [batch, 28] (explained in Attr.py)

        # sptm_s: hidden sequence (B * T * F); sptm_l: lens (list of int); sptm_t: merged tensor after attention/mean pooling
        ### sptm_s size [batch, variable (packed), 128], sptm_s size [batch, ], lens size [batch, 128]
        sptm_s, sptm_l, sptm_t = self.spatio_temporal(traj, attr_t, config)

        entire_out = self.entire_estimate(attr_t, sptm_t)   ### size [batch, 1]

        # sptm_s is a packed sequence (see pytorch doc for details), only used during the training
        if self.training:
            ### sptm_s[0] refers to values from all time steps concatenated together into a single dim, size [variable, 128]
            local_out = self.local_estimate(sptm_s[0])   ### size [variable, 1]
            return entire_out, (local_out, sptm_l)
        else:
            return entire_out

    def batch_step(self, data):
        attr = data[0]
        traj = data[1]
        for key in attr.keys():
            attr[key] = attr[key].to(self.device)
        for key in traj.keys():
            if key != 'lens':
                traj[key] = traj[key].to(self.device)
        config = self.config

        if self.training:
            entire_out, (local_out, local_length) = self(attr, traj, config)
        else:
            entire_out = self(attr, traj, config)

        pred_dict, entire_loss = self.entire_estimate.eval_on_batch(entire_out, attr['time'], config['time_mean'], config['time_std'])

        if self.training:
            # get the mean/std of each local path
            mean, std = (self.kernel_size - 1) * config['time_gap_mean'], (self.kernel_size - 1) * config['time_gap_std']
            # get ground truth of each local path
            local_label = deeptte_utils.get_local_seq(traj['time_gap'], self.kernel_size, mean, std)
            local_loss = self.local_estimate.eval_on_batch(local_out, local_length, local_label, mean, std)

            return pred_dict['label'], pred_dict['pred'], (1 - self.alpha) * entire_loss + self.alpha * local_loss   ### According to eqn 8 of paper
        else:
            return pred_dict['label'], pred_dict['pred'], entire_loss

    def evaluate(self, test_dataloader, config):
        labels, preds, avg_batch_loss = model_utils.predict(self, test_dataloader)
        return labels, preds