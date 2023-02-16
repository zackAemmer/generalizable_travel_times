import numpy as np
import pandas as pd

from database import data_utils


class TimeTableModel:
    def __init__(self, gtfs_data):
        self.gtfs_data = gtfs_data

    def predict_simple_sch(self, data):
        trip_ids = [x['trip_id'] for x in data]
        lons = [x['lngs'][-1] for x in data]
        lats = [x['lats'][-1] for x in data]
        _, arrival_s_from_midnight, _, _ = data_utils.get_scheduled_arrival(trip_ids, lons, lats, self.gtfs_data)
        current_s_from_midnight = np.array([x['timeID_s'][0] for x in data])
        preds = arrival_s_from_midnight - current_s_from_midnight
        return preds

    def save_to(self, folder, name):
        data_utils.write_pkl(self, folder + name + ".pkl")
        return None