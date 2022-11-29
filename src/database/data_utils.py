#!/usr/bin/env python3
"""Functions for processing and working with tracked bus data.
"""


from datetime import datetime, timezone, timedelta
import json
import os
import requests
from zipfile import ZipFile

import numpy as np
import pandas as pd


def get_validation_dates(validation_path):
    dates = []
    files = os.listdir(validation_path)
    for file in files:
        labels = file.split("-")
        dates.append(labels[2] + "-" + labels[3] + "-" + labels[4].split("_")[0])
    return dates

def extract_validation_trips(validation_path):
    files = os.listdir(validation_path)
    for file in files:
        labels = file.split("-")
        vehicle_id = labels[0]
        route_num = labels[1]
        year = labels[2]
        month = labels[3]
        day = labels[4].split("_")[0]
        hour = labels[4].split("_")[1]
        minute = labels[5]
        day_data = pd.read_csv(f"./data/kcm_validation_tracks/{labels}")
    # Get the tracks for the route id and time of the validation data +/- amount

    return vehicle_id
