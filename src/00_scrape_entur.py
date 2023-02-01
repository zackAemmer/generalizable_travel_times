#!/usr/bin/env python3
import requests
from datetime import datetime, timedelta
import re
import time
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import psycopg2

# import config as secret


# def connect_to_rds():
#     conn = psycopg2.connect(
#         host=secret.HOST,
#         database=secret.DATABASE,
#         user=secret.UID,
#         password=secret.PWD)
#     return conn

def get_epoch_and_cet_24hr():
    # Get the UTC/GMT epoch, and the CET hour of day
    utc = datetime.utcnow()
    cet = timedelta(hours=1)
    current_hour = (utc + cet).hour
    epoch = round(utc.timestamp())
    return current_hour, epoch

def datetime_to_epoch(time):
    # Go from time string in CET to epoch in UTC/GMT
    # format is '2022-03-16T16:14:58.12+01:00'
    yr = int(time[0:4])
    mo = int(time[5:7])
    day = int(time[8:10])
    hr = int(time[11:13])
    mn = int(time[14:16])
    sec = int(time[17:19])
    ts= datetime(yr, mo, day, hr, mn, sec).timestamp()
    return ts - (60.0 * 60)

def xml_to_dict(element):
    # Recursively create a dictionary of XML field -> text value
    element_dict = {}
    for child in element:
        tag = re.split("}", child.tag)[1]
        if child.text != None:
            element_dict[tag] = child.text
        elif tag in element_dict.keys(): # In case multiple children with same tag exist in this element, turn into a list
            if type(element_dict[tag]) == list:
                element_dict[tag].append(xml_to_dict(child))
            else:
                first_elem = element_dict[tag]
                element_dict[tag] = []
                element_dict[tag].append(first_elem)
        else:
            element_dict[tag] = xml_to_dict(child)
    return element_dict

def exists_str_or_none(values, key):
    # Look for key in values dict. Return string if exists, otherwise None.
    if key in values.keys():
        if key == 'OnwardCalls':
            return exists_str_or_none(values[key], 'OnwardCall')
        elif key == 'OnwardCall' and type(values[key]) == list:
            return str(values[key][0]['StopPointRef'])
        elif key == 'OnwardCall' and type(values[key]) == dict:
            return str(values[key]['StopPointRef'])
        else:
            return str(values[key])
    else:
        return None

def clean_active_trips(vehicle_statuses):
    # Preprocess the trip status data; keep as much as possible but remove if no GPS data.
    to_remove = []
    necessary_keys = ['FramedVehicleJourneyRef','VehicleLocation'] # The absolute minimum for useful data point
    # Find indices of trips that are not monitored.
    for i, vehicle in enumerate(vehicle_statuses):
        try:
            if vehicle['MonitoredVehicleJourney']['Monitored'] != 'true': # These keys are implicitly necessary as well
                to_remove.append(i)
                continue
            elif not 'RecordedAtTime' in vehicle.keys(): # Using LocationRecordedAtTime leads to more data loss
                to_remove.append(i)
                continue
            else:
                for key in necessary_keys:
                    if not key in vehicle['MonitoredVehicleJourney'].keys():
                        to_remove.append(i)
                        break
        # If the requested value isn't found, except and remove
        except:
            to_remove.append(i)
    # Remove inactive trips starting with the last index to avoid messing up indices as it goes
    for idx in sorted(to_remove, reverse=True):
        del vehicle_statuses[idx]
    return vehicle_statuses

def upload_to_rds(to_upload, conn, collected_time):
    to_upload_list = []
    for vehicle_status in to_upload:
        try:
            datedvehiclejourney = exists_str_or_none(vehicle_status['MonitoredVehicleJourney']['FramedVehicleJourneyRef'], 'DatedVehicleJourneyRef'), # JourneyPatternRef[1:3]-VehicleJourneyRef-??-JourneyPatternRef[4:6]-YYYYMMDD-????
            dataframe = exists_str_or_none(vehicle_status['MonitoredVehicleJourney']['FramedVehicleJourneyRef'], 'DataFrameRef'),
            vehicle = exists_str_or_none(vehicle_status['MonitoredVehicleJourney'], 'VehicleRef'),
            mode = exists_str_or_none(vehicle_status['MonitoredVehicleJourney'], 'VehicleMode'),
            line = exists_str_or_none(vehicle_status['MonitoredVehicleJourney'], 'LineRef'),
            linename = exists_str_or_none(vehicle_status['MonitoredVehicleJourney'], 'PublishedLineName'),
            direction = exists_str_or_none(vehicle_status['MonitoredVehicleJourney'], 'DirectionRef'),
            operator = exists_str_or_none(vehicle_status['MonitoredVehicleJourney'], 'OperatorRef'),
            datasource = exists_str_or_none(vehicle_status['MonitoredVehicleJourney'], 'DataSource'),
            lat = exists_str_or_none(vehicle_status['MonitoredVehicleJourney']['VehicleLocation'], 'Latitude'),
            lon = exists_str_or_none(vehicle_status['MonitoredVehicleJourney']['VehicleLocation'], 'Longitude'),
            bearing = exists_str_or_none(vehicle_status['MonitoredVehicleJourney'], 'Bearing'),
            delay = exists_str_or_none(vehicle_status['MonitoredVehicleJourney'], 'Delay'),
            nextstop = exists_str_or_none(vehicle_status['MonitoredVehicleJourney'], 'OnwardCalls'),
            locationtime = datetime_to_epoch(exists_str_or_none(vehicle_status, 'RecordedAtTime')),
            collectedtime = collected_time
            to_upload_list.append((datedvehiclejourney, dataframe, vehicle, mode, line, linename, direction, operator, datasource, lat, lon, bearing, delay, nextstop, locationtime, collectedtime))
        except Exception as e:
            print("Error: Data Formatting")
            print(vehicle_status)
            print(e)
    with conn.cursor() as curs:
        try:
            args_str = ','.join(curs.mogrify('(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', x).decode('utf-8') for x in to_upload_list)
            query_str = 'INSERT INTO active_trips_norway (datedvehiclejourney, dataframe, vehicle, mode, line, linename, direction, operator, datasource, lat, lon, bearing, delay, nextstop, locationtime, collectedtime) VALUES ' + args_str
            curs.execute(query_str)
            conn.commit()
        except Exception as e:
            print("Error: Upload Rollback")
            print(e)
            conn.rollback()
    return query_str

if __name__ == "__main__":
    # Limited to 4 requests/minute, otherwise need publish/subscribe
    endpoint = 'https://api.entur.io/realtime/v1/rest/vm'
    # conn = connect_to_rds()
    current_hour, current_epoch = get_epoch_and_cet_24hr()
    while current_hour < 19:
        # Call Entur SIRI API (returns XML)
        response = requests.get(endpoint)
        root = ElementTree.fromstring(response.content)
        # root = ElementTree.parse('vm.xml').getroot() # For testing without hitting API

        # Look at list of active vehicles from response
        root_dict = xml_to_dict(root)
        vehicle_statuses = root_dict['ServiceDelivery']['VehicleMonitoringDelivery']['VehicleActivity']
        clean_active_trips(vehicle_statuses) # Modifies in-place by deleting elements to save memory

        current_hour, current_epoch = get_epoch_and_cet_24hr()
        # args_str = upload_to_rds(vehicle_statuses, conn, current_epoch)
        time.sleep(20)
    # conn.close()
