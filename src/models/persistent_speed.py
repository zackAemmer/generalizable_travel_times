import numpy as np

from utils import data_utils


class PersistentSpeedSeqModel:
    def __init__(self, config, min_speed):
        self.config = config
        self.min_speed = min_speed
        return None

    def predict(self, dataloader):
        context, X, y, seq_lens = data_utils.extract_all_dataloader(dataloader, sequential_flag=True)
        max_len = max(seq_lens)
        # Carry the starting speeds through full sequences
        preds = X[:,0,5].unsqueeze(1).repeat((1,max_len)).numpy()
        preds = data_utils.de_normalize(preds, self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        # Assume a minimum speed, otherwise infinite/exploding travel times are possible (0 m/s)
        preds[preds<self.min_speed] = self.min_speed
        labels = y.numpy()
        labels = data_utils.de_normalize(labels, self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        return labels, preds

    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None