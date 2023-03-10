import numpy as np
import pandas as pd

from utils import data_utils


class TimeTableModel:
    def __init__(self, config):
        # Saving the data itself takes up too much space
        self.config = config
        self.gtfs_folder = config['gtfs_folder']

    def predict_simple_sch(self, dataloader):
        data, labels = dataloader.dataset.tensors
        # The scheduled arrival times, and current times, are already features in the data
        arrival_s_from_midnight = np.array([x[0][3].numpy() for x in data])
        arrival_s_from_midnight = data_utils.de_normalize(arrival_s_from_midnight, self.config['scheduled_time_s_mean'], self.config['scheduled_time_s_std'])
        current_s_from_midnight = np.array([x[1][0].numpy() for x in data])
        preds = arrival_s_from_midnight - current_s_from_midnight
        labels = data_utils.de_normalize(labels.numpy(), self.config['time_mean'], self.config['time_std'])
        return labels, preds

    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None