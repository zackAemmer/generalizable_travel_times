import numpy as np
import pandas as pd

from utils import data_utils


class AvgHourlySpeedModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.speed_lookup = {}
        return None
    def fit(self, dataloader, config):
        context, X, y = data_utils.extract_all_dataloader(dataloader)
        speeds = X[:,8].numpy()
        speeds = data_utils.de_normalize(speeds, config['speed_m_s_mean'], config['speed_m_s_std'])
        hours = context[:,0].numpy() // 60
        # Calculate average speed grouped by time of day
        self.speed_lookup = pd.DataFrame({"hour":hours, "speed":speeds}).groupby("hour").mean().to_dict()
        return None
    def predict(self, dataloader, config):
        context, X, y = data_utils.extract_all_dataloader(dataloader)
        hours = context[:,0].numpy() // 60
        speeds = [self.get_speed_if_available(x) for x in hours]
        dists = X[:,10].numpy()
        dists = data_utils.de_normalize(dists, config['dist_mean'], config['dist_std'])
        preds = dists*1000.0 / speeds
        labels = y.numpy()
        labels = data_utils.de_normalize(labels, config['time_mean'], config['time_std'])
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