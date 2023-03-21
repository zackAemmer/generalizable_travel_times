import numpy as np

from utils import data_utils


class TimeTableModel:
    def __init__(self, model_name, config):
        self.model_name = model_name
        # Saving the data itself takes up too much space
        self.config = config
        self.gtfs_folder = config['gtfs_folder']

    def predict(self, dataloader):
        context, X, y = data_utils.extract_all_dataloader(dataloader)
        scheduled_time_s = X[:,4].numpy()
        scheduled_time_s = data_utils.de_normalize(scheduled_time_s, self.config['scheduled_time_s_mean'], self.config['scheduled_time_s_std'])
        preds = scheduled_time_s
        labels = y.numpy()
        labels = data_utils.de_normalize(labels, self.config['time_mean'], self.config['time_std'])
        return labels, preds

    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None