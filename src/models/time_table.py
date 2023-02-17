import numpy as np
import pandas as pd

from database import data_utils


class TimeTableModel:
    def __init__(self, gtfs_folder):
        # Saving the data itself takes up too much space
        self.gtfs_folder = gtfs_folder

    def predict_simple_sch(self, data, gtfs_data):
        trip_ids = [x['trip_id'] for x in data]
        lons = [x['lngs'][-1] for x in data]
        lats = [x['lats'][-1] for x in data]
        _, arrival_s_from_midnight, _, _ = data_utils.get_scheduled_arrival(trip_ids, lons, lats, gtfs_data)
        current_s_from_midnight = np.array([x['timeID_s'][0] for x in data])
        preds = arrival_s_from_midnight - current_s_from_midnight
        return preds

    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None