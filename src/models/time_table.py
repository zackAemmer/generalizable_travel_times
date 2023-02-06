import geopandas
import numpy as np
import pandas as pd
import shapely.geometry

from database import data_utils


class TimeTableModel:
    def __init__(self, gtfs_folder, timezone):
        self.gtfs_folder = gtfs_folder
        self.timezone = timezone
        self.gtfs_data = data_utils.merge_gtfs_files(self.gtfs_folder)

    def predict_using_schedule_only(self, traces):
        # Calculate baseline for test data from GTFS timetables
        # If the trip is not in the time tables, we cannot make a prediction
        traces_cp = traces[traces['trip_id'].isin(self.gtfs_data['trip_id'])].copy()
        num_lost = (len(traces) - len(traces_cp)) / len(traces)
        # Get the present time for each point
        locations_dt = pd.to_datetime(traces_cp['locationtime'], unit="s", utc=True).map(lambda x: x.tz_convert(self.timezone))
        traces_cp['actual_time_from_midnight'] = locations_dt.dt.hour*60*60 + locations_dt.dt.minute*60 + locations_dt.dt.second
        # Get first/last point of each trace
        start_locations = traces_cp.groupby(["file",'trip_id']).nth([0]).reset_index()
        end_locations = traces_cp.groupby(["file",'trip_id']).nth([-1]).reset_index()
        # Cross join all endpoints to all stops that trip uses
        df = pd.merge(end_locations, self.gtfs_data, left_on=['trip_id'], right_on=["trip_id"])
        # Group by file/trip and keep only the minimum distance stop
        df['dist_to_stop'] = data_utils.calculate_gps_dist(df.lon, df.lat, df.stop_lon, df.stop_lat)
        df = df.sort_values(['file','trip_id','dist_to_stop'])
        df = df.groupby(["file",'trip_id']).nth([0]).reset_index()
        # Calculate the arrival time in seconds since midnight for closest stop
        df['arrival_s'] = [int(x[0])*60*60 + int(x[1])*60 + int(x[2]) for x in df['arrival_time'].str.split(":")]
        # The predicted travel time starts at the trace start, and goes until the arrival at the nearest stop
        preds = df['arrival_s'] - start_locations['actual_time_from_midnight']
        # The actual travel time ends at exactly when the final trace was taken
        labels = df['actual_time_from_midnight'] - start_locations['actual_time_from_midnight']
        return preds, labels, num_lost

    def predict_using_shapefiles(self, traces):
        # Join the GTFS info to the trace data
        traces = pd.merge(traces, self.gtfs_data, left_on=["tripid","nextstop"], right_on=["trip_id","stop_id"])
        # Split out the start and end locations of each trajectory
        start_locations = traces.groupby(["file","tripid"]).nth([0]).reset_index()
        end_locations = traces.groupby(["file","tripid"]).nth([-1]).reset_index()
        start_locations = self.calculate_scheduled_time_of_day(start_locations)
        end_locations = self.calculate_scheduled_time_of_day(end_locations)
        # Predict the travel times
        preds = end_locations['predicted_time_from_midnight'] - start_locations['actual_time_from_midnight']
        labels = end_locations['actual_time_from_midnight'] - start_locations['actual_time_from_midnight']
        return preds, labels

    def plot_gtfs_trip(self, trip_id, ax):
        z = self.gtfs_data
        z = z[z['trip_id'] == trip_id]
        stop_geometry = [shapely.Point(x,y) for x,y in zip(z.stop_lon, z.stop_lat)]
        stop_geometry = geopandas.GeoDataFrame(crs='epsg:4326', geometry=stop_geometry)
        z.plot(ax=ax)
        stop_geometry.plot(ax=ax)
        return None

    def plot_trace_and_nextstop(self, traces, trip_id, file_id, ax):
        z = traces[traces['file']==file_id]
        z = z[z['tripid']==trip_id]
        z = geopandas.GeoDataFrame(z)
        stop_geometry = [shapely.Point(x,y) for x,y in zip(z.stop_lon, z.stop_lat)]
        stop_geometry = geopandas.GeoDataFrame(crs='epsg:4326', geometry=stop_geometry)
        z.plot(ax=ax)
        stop_geometry.plot(ax=ax, marker="x")
        return z

    def shapes_to_segments(self, gtfs_folder):
        shapes = pd.read_csv(gtfs_folder+"shapes.txt")
        shapes = geopandas.GeoDataFrame(shapes, geometry=geopandas.points_from_xy(shapes.shape_pt_lon, shapes.shape_pt_lat))
        segment_shapes = shapes.groupby(['shape_id'])['geometry'].apply(lambda x: shapely.geometry.LineString(x.tolist()))
        segment_shapes = geopandas.GeoDataFrame(segment_shapes, geometry='geometry', crs="EPSG:4326")
        segment_shapes.reset_index(inplace=True)
        return segment_shapes

    def process_stop_times(self, gtfs_folder):
        stop_times = pd.read_csv(gtfs_folder+"stop_times.txt")
        # Need to know info for both the upcoming stop, and its previous stop for interpolation of bus position
        stop_times = stop_times.sort_values(['trip_id', 'stop_sequence'])
        prev_stop_times = stop_times.shift(1)
        prev_stop_times.columns = [x+"_prev" for x in prev_stop_times.columns]
        stop_times = pd.concat([stop_times, prev_stop_times], axis=1)
        # Cannot do first stop since there is no previous. In any case a trip where nextstop is the initial stop should be filtered from the data.
        stop_times = stop_times[stop_times['trip_id']==stop_times['trip_id_prev']]
        # Convert H:M:S to seconds of the day
        stop_times['arrival_s'] = [int(x[0])*60*60 + int(x[1])*60 + int(x[2]) for x in stop_times['arrival_time'].str.split(":")]
        stop_times['arrival_s_prev'] = [int(x[0])*60*60 + int(x[1])*60 + int(x[2]) for x in stop_times['arrival_time_prev'].str.split(":")]
        # Calculate the speed determined by the schedule between each consecutive stop
        stop_times['dist_diff'] = stop_times['shape_dist_traveled'] - stop_times['shape_dist_traveled_prev']
        stop_times['time_diff'] = stop_times['arrival_s'] - stop_times['arrival_s_prev']
        stop_times['speed_m_s_scheduled'] = stop_times['dist_diff'] / stop_times['time_diff']
        return stop_times

    def calculate_scheduled_time_of_day(self, locations):
        # Get geometry of ending points
        point_geometry = [shapely.Point(x,y) for x,y in zip(locations.lon, locations.lat)]
        point_geometry = geopandas.GeoDataFrame(crs='epsg:4326', geometry=point_geometry)
        # Get projection onto GTFS shape of final trajectory location
        locations['dist_along_shape'] = [x.project(y, normalized=False)*111000 for x,y in zip(locations.geometry, point_geometry.geometry)]
        locations['dist_to_nextstop'] = locations['shape_dist_traveled'] - locations['dist_along_shape']
        # Get the scheduled quantity of time until the bus arrives at the next stop
        locations['time_to_nextstop'] = locations['dist_to_nextstop'] / locations['speed_m_s_scheduled']
        # Get the expected current time, if the bus is to be at the nextstop at its scheduled time
        locations['predicted_time_from_midnight'] = locations['arrival_s'] - locations['time_to_nextstop']
        # Get the difference between the realtime observation measured from midnight, and the projected time based on the schedule
        locations_dt = pd.to_datetime(locations['locationtime'], unit="s", utc=True).map(lambda x: x.tz_convert(self.timezone))
        locations['actual_time_from_midnight'] = locations_dt.dt.hour*60*60 + locations_dt.dt.minute*60 + locations_dt.dt.second
        return locations
