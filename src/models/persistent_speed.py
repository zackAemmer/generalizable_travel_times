import numpy as np
import pandas as pd
import torch

from utils import data_utils

class PersistentSpeedSeqModel:
    def __init__(self, config, seq_len):
        self.config = config
        self.seq_len = seq_len
        return None

    def predict(self, dataloader):
        data, labels = dataloader.dataset.tensors
        # Predict that next speed will be same as previous speed
        speeds = np.array([x[0][self.seq_len-1,5].numpy() for x in data], dtype='float32')        
        preds = data_utils.de_normalize(speeds, self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        labels = data_utils.de_normalize(labels.numpy().flatten(), self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        return labels, preds

    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None