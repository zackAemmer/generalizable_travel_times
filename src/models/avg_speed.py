import numpy as np
import pandas as pd

from utils import data_utils, data_loader


class AvgHourlySpeedModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.speed_lookup = {}
        self.requires_grid = False
        self.collate_fn = data_loader.basic_collate_nosch
        self.train_time = 0.0
        self.hyperparameter_dict = {'BATCH_SIZE': 512}
        self.is_nn = False
        return None
    def train(self, dataloader, config):
        data = [x for x in dataloader]
        speeds = data_utils.de_normalize(np.concatenate([x[0][1] for x in data])[:,4], config['speed_m_s_mean'], config['speed_m_s_std'])
        hours = np.concatenate([x[0][0] for x in data])[:,0] // 60
        # Calculate average speed grouped by time of day
        self.speed_lookup = pd.DataFrame({"hour":hours, "speed":speeds}).groupby("hour").mean().to_dict()
        return None
    def evaluate(self, dataloader, config):
        data = [x for x in dataloader]
        hours = np.concatenate([x[0][0] for x in data])[:,0] // 60
        dists = data_utils.de_normalize(np.concatenate([x[0][1] for x in data])[:,6], config['dist_mean'], config['dist_std'])
        speeds = np.array([self.get_speed_if_available(x) for x in hours])
        preds = dists*1000.0 / speeds
        labels = data_utils.de_normalize(np.concatenate([x[1] for x in data]), config['time_mean'], config['time_std'])
        return labels, preds
    def get_speed_if_available(self, hour):
        # If no data was available for the requested hour, return the mean of all available hours
        if hour in self.speed_lookup['speed'].keys():
            speed = self.speed_lookup['speed'][hour]
            # If there is an hour with 0.0 speeds due to small sample size it will cause errors
            if speed == 0.0:
                return np.mean(list(self.speed_lookup['speed'].values()))
            else:
                return speed
        else:
            return np.mean(list(self.speed_lookup['speed'].values()))
    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None