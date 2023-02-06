import numpy as np
import pandas as pd


class AvgHourlySpeedModel:
    def __init__(self):
        return None

    def fit(self, traces):
        # Calculate average speed grouped by time of day
        dists = [x['dist'] for x in traces]
        times = [x['time'] for x in traces]
        hours = [x['timeID'] // 60 for x in traces]
        speeds = [dists[i] / times[i] for i in range(0,len(dists))] # km/s
        self.speed_lookup = pd.DataFrame({"hour":hours, "speed":speeds}).groupby("hour").mean().to_dict()
        return None

    def predict(self, traces):
        # Predict travel time based on historical average speeds
        hours = [x['timeID'] // 60 for x in traces]
        dists = [x['dist'] for x in traces]
        speeds = [self.get_speed_if_available(x) for x in hours]
        preds = [dists[i] / speeds[i] for i in range(0,len(dists))]
        return preds

    def get_speed_if_available(self, hour):
        if hour in self.speed_lookup.keys():
            return self.speed_lookup['speed'][hour]
        else:
            return np.mean(list(self.speed_lookup['speed'].values()))
