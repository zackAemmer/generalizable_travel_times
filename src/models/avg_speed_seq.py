import numpy as np
import pandas as pd
import torch

from utils import data_utils

class AvgHourlySpeedSeqModel:
    def __init__(self, config, seq_len):
        self.speed_lookup = {}
        self.config = config
        self.seq_len = seq_len
        return None

    def fit(self, dataloader):
        data, labels = dataloader.dataset.tensors
        # Use average of first speed across all sequences
        speeds = np.array([x[0][0,5].numpy() for x in data])
        speeds = data_utils.de_normalize(speeds, self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        hours = np.array([x[1][0].numpy() for x in data]) // 60
        # Calculate average speed grouped by time of day
        self.speed_lookup = pd.DataFrame({"hour":hours, "speed":speeds}).groupby("hour").mean().to_dict()
        return None

    def predict(self, dataloader):
        data, labels = dataloader.dataset.tensors
        # Predict travel time based on historical average speeds
        hours = np.array([x[1][0].numpy() for x in data]) // 60
        preds = np.array([self.get_speed_if_available(x) for x in hours], dtype='float32')
        labels = data_utils.de_normalize(labels.numpy(), self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        return labels, preds

    def get_speed_if_available(self, hour):
        # If no data was available for the requested hour, return the mean of all available hours
        if hour in self.speed_lookup['speed'].keys():
            return self.speed_lookup['speed'][hour]
        else:
            return np.mean(list(self.speed_lookup['speed'].values()))

    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None