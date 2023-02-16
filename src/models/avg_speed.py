import numpy as np
import pandas as pd

from database import data_utils

class AvgHourlySpeedModel:
    def __init__(self):
        self.speed_lookup = {}
        return None

    def fit(self, data):
        # Calculate average speed grouped by time of day
        dists = [x['dist'] for x in data]
        times = [x['time'] for x in data]
        hours = [x['timeID'] // 60 for x in data]
        speeds = [dists[i] / times[i] for i in range(0,len(dists))] # km/s
        self.speed_lookup = pd.DataFrame({"hour":hours, "speed":speeds}).groupby("hour").mean().to_dict()
        return None

    def predict(self, data):
        # Predict travel time based on historical average speeds
        hours = [x['timeID'] // 60 for x in data]
        dists = [x['dist'] for x in data]
        speeds = [self.get_speed_if_available(x) for x in hours]
        preds = np.array([dists[i] / speeds[i] for i in range(0,len(dists))])
        return preds

    def get_speed_if_available(self, hour):
        # If no data was available for the requested hour, return the mean of all available hours
        if hour in self.speed_lookup.keys():
            return self.speed_lookup['speed'][hour]
        else:
            return np.mean(list(self.speed_lookup['speed'].values()))

    def save_to(self, folder, name):
        data_utils.write_pkl(self, folder + name + ".pkl")
        return None