#!/usr/bin/env python3
import requests
from datetime import datetime, timedelta
import time
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

import config as cfg


def connect_to_rds():
    conn = psycopg2.connect(
        host=cfg.HOST,
        database=cfg.DATABASE,
        user=cfg.UID,
        password=cfg.PWD)
    return conn

def get_epoch_and_cet_24hr():
    utc = datetime.utcnow()
    cet = timedelta(hours=1)
    current_hour = (utc + cet).hour
    epoch = round(utc.timestamp())
    return current_hour, epoch

def remove_agency_tag(id_string):
    underscore_loc = id_string.find('_')
    final = int(id_string[underscore_loc+1:])
    return(final)

def clean_active_trips(response):
    active_trip_statuses = response['data']['list']
    to_remove = []
    # Find indices of trips that are inactive or have no data
    for i, bus in enumerate(response['data']['list']):
        if bus['tripId'] == '' or bus['status'] == 'CANCELED' or bus['tripStatus']['position'] == None:
            to_remove.append(i)
    # Remove inactive trips starting with the last index
    for index in sorted(to_remove, reverse=True):
        del active_trip_statuses[index]
    return active_trip_statuses

def upload_to_rds(to_upload, conn, collected_time):
    to_upload_list = []
    for bus_status in to_upload:
        to_upload_list.append(
            (str(remove_agency_tag(bus_status['tripId'])),
            str(remove_agency_tag(bus_status['vehicleId'])),
            str(round(bus_status['tripStatus']['position']['lat'], 10)),
            str(round(bus_status['tripStatus']['position']['lon'], 10)),
            str(round(bus_status['tripStatus']['orientation'])),
            str(bus_status['tripStatus']['scheduleDeviation']),
            str(round(bus_status['tripStatus']['totalDistanceAlongTrip'], 10)),
            str(round(bus_status['tripStatus']['distanceAlongTrip'], 10)),
            str(remove_agency_tag(bus_status['tripStatus']['closestStop'])),
            str(remove_agency_tag(bus_status['tripStatus']['nextStop'])),
            str(bus_status['tripStatus']['lastLocationUpdateTime'])[:-3],
            str(collected_time)))
    with conn.cursor() as curs:
        try:
            args_str = ','.join(curs.mogrify('(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', x).decode('utf-8') for x in to_upload_list)
            query_str = 'INSERT INTO active_trips_study (tripid, vehicleid, lat, lon, orientation, scheduledeviation, totaltripdistance, tripdistance, closeststop, nextstop, locationtime, collectedtime) VALUES ' + args_str
            curs.execute(query_str)
            conn.commit()
        except:
            # Catch all errors and continue to keep server up and running
            conn.rollback()
    return query_str

def main_function():
    # Limited to 4 requests/minute, otherwise need publish/subscribe
    endpoint = 'https://api.entur.io/realtime/v1/rest/vm'
    conn = connect_to_rds()
    current_hour, current_epoch = get_epoch_and_cet_24hr()
    while True or current_hour < 19:
        # response = requests.get(endpoint)
        # tree = ElementTree.fromstring(response.content)
        tree = ElementTree.parse('vm.xml')
        root = tree.getroot()
        for child in root[0][0]:
            print(child.tag, child.attrib)
        current_hour, current_epoch = get_epoch_and_cet_24hr()
        # cleaned_response = clean_active_trips(response)
        # args_str = upload_to_rds(cleaned_response, conn, current_epoch)
        time.sleep(18)
    conn.close()

if __name__ == "__main__":
    main_function()
