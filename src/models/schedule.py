import numpy as np

from utils import data_utils, data_loader


class TimeTableModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.requires_grid = False
        self.collate_fn = data_loader.basic_collate
        self.train_time = 0.0
        self.hyperparameter_dict = {'BATCH_SIZE': 512}
        self.is_nn = False
    def train(self, dataloader, config):
        return None
    def evaluate(self, dataloader, config):
        data = [x for x in dataloader]
        preds = data_utils.de_normalize(np.concatenate([x[0][1] for x in data])[:,5], config['scheduled_time_s_mean'], config['scheduled_time_s_std'])
        labels = data_utils.de_normalize(np.concatenate([x[1] for x in data]), config['time_mean'], config['time_std'])
        return labels, preds
    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None