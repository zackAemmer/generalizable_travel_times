import numpy as np

from utils import data_utils


class TimeTableModel:
    def __init__(self, config):
        # Saving the data itself takes up too much space
        self.config = config
        self.gtfs_folder = config['gtfs_folder']

    def predict_simple_sch(self, dataloader):
        data, labels = dataloader.dataset.tensors
        # The scheduled arrival times, and current times, are already features in the data
        scheduled_time_s = np.array([x[0][4].numpy() for x in data])
        scheduled_time_s = data_utils.de_normalize(scheduled_time_s, self.config['scheduled_time_s_mean'], self.config['scheduled_time_s_std'])
        preds = scheduled_time_s
        labels = data_utils.de_normalize(labels.numpy(), self.config['time_mean'], self.config['time_std'])
        return labels, preds

    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None