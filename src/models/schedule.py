import numpy as np

from utils import data_utils, data_loader


class TimeTableModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.requires_grid = False
        self.collate_fn = data_loader.basic_collate
        self.train_time = 0.0
        self.is_nn = False
    def train(self, dataloader, config):
        return None
    def evaluate(self, dataloader, config):
        context, X, y = data_utils.extract_all_dataloader(dataloader)
        scheduled_time_s = X[:,5].numpy()
        scheduled_time_s = data_utils.de_normalize(scheduled_time_s, config['scheduled_time_s_mean'], config['scheduled_time_s_std'])
        preds = scheduled_time_s
        labels = y.numpy()
        labels = data_utils.de_normalize(labels, config['time_mean'], config['time_std'])
        return labels, preds
    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None