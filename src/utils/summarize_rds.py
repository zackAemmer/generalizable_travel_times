import os
import pickle
from datetime import datetime
from zipfile import ZipFile

import numpy as np
import pandas as pd
import psycopg2
import pytz
import requests
from sklearn.neighbors import BallTree


def convert_cursor_to_tabular(query_result_cursor):
    """Converts a cursor returned by a SQL execution to a Pandas dataframe.

    First iterates through the cursor and dumps all of the contents into a numpy
    array for easier access. Then, a dataframe is created and grown to store the
    full contents of the cursor. This function's main purpose is to make the
    query results easier to work with in other functions. It may slow down the
    processing especially if extremely large (>24hrs) queries are made.

    Args:
        query_result_cursor: A Psycopg Cursor object pointing to the first
            result from a query for bus locations from the data warehouse. The
            results should contain columns for tripid, vehicleid, orientation,
            scheduledeviation, closeststop, nextstop, locationtime, and
            collectedtime.

    Returns:
        A Pandas Dataframe object containing the query results in tabular
        form.
    """
    # Pull out all the variables from the query result cursor and store in array
    all_tracks = []
    for record in query_result_cursor:
        track = []
        for feature in record:
            track = np.append(track, feature)
        all_tracks.append(track)

    # Convert variables integers, store in Pandas, and return
    daily_results = pd.DataFrame(all_tracks)
    colnames = []
    for col in query_result_cursor.description:
        colnames.append(col.name)
    # If not enough/no data was recorded on the day of interest this will return
    if len(daily_results.columns) == 0:
        return None
    else:
        daily_results.columns = colnames
    daily_results = daily_results.dropna()
    return daily_results

def connect_to_rds():
    """Connects to the RDS data warehouse specified in config.py.

    Attempts to connect to the database, and if successful it returns a
    connection object that can be used for queries to the bus location data.

    Returns:
        A Psycopg Connection object for the RDS data warehouse specified in
        config.py.
    """
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        port=os.getenv('DB_PORT'),
        user=os.getenv('DB_UID'),
        password=os.getenv('DB_PASSWORD'))
    return conn

def get_results_by_time(conn, start_time, end_time, rds_limit, table_name, unique_trip_col):
    """Queries the last x days worth of data from the RDS data warehouse.

    Uses the database connection to execute a query for the specified times of
    bus coordinates stored in the RDS data warehouse. The RDS data must have a
    column for collected time (in epoch format) which is used to determine the
    time. All time comparisons between the RDS and the system are done in epoch
    time, so there should be no concern for time zone differences if running this
    function from an EC2 instance.

    Args:
        conn: A Psycopg Connection object for the RDS data warehouse.
        start_time: An integer specifying the start of the range of times that
            should be collected from the database.
        end_time: An integer specifying the end of the range of times that should
            be collected from the database.
        rds_limit: An integer specifying the maximum number of rows to query.
            Useful for debugging and checking output before making larger
            queries. Set to 0 for no limit.

    Returns:
        A Pandas Dataframe object containing the results in the database for the
        last x day period.
    """
    # Database has index on locationtime attribute
    if rds_limit > 0:
        query_text = f" \
            SELECT DISTINCT ON ({unique_trip_col}, locationtime) * \
            FROM {table_name} \
            WHERE locationtime \
            BETWEEN {start_time} AND {end_time} \
            ORDER BY {unique_trip_col}, locationtime, collectedtime ASC \
            LIMIT {rds_limit};"
    else:
        query_text = f" \
            SELECT DISTINCT ON ({unique_trip_col}, locationtime) * \
            FROM {table_name} \
            WHERE locationtime \
            BETWEEN {start_time} AND {end_time} \
            ORDER BY {unique_trip_col}, locationtime, collectedtime ASC;"
    with conn.cursor() as curs:
        curs.execute(query_text)
        daily_results = convert_cursor_to_tabular(curs)
    return daily_results

def update_gtfs_route_info():
    """Downloads the latest trip-route conversions from the KCM GTFS feed.

    Connects to the King County Metro GTFS server and requests the latest GTFS
    files. Saves the files in .zip format and then extracts their content to a
    folder named 'google transit'. This will be used when assigning route ids to
    the bus coordinate data from RDS, so that they can then be aggregated to
    matching segments.

    Returns:
        A string with the location of the folder where the GTFS data is saved.
    """
    url = 'http://metro.kingcounty.gov/GTFS/google_transit.zip'
    zip_location = './data/kcm_gtfs'
    req = requests.get(url, allow_redirects=True)
    with open('./data/kcm_gtfs/google_transit.zip', 'wb') as g_file:
        g_file.write(req.content)
    with ZipFile('./data/kcm_gtfs/google_transit.zip', 'r') as zip_obj:
        zip_obj.extractall('./data/kcm_gtfs')
    return zip_location

def preprocess_trip_data(daily_results):
    """Cleans the tabular trip data and calculates average speed.

    Removes rows with duplicated tripid, locationid columns from the data. These
    rows are times where the OneBusAway API was queried faster than the bus
    location was updated, creating duplicate info. Buses update at around 30s
    intervals and the API is queried at 10s intervals so there is a large amount
    of duplicate data. Speeds are calculated between consecutive bus locations
    based on the distance traveled and the time between those locations. Speeds
    that are below 0 m/s, or above 30 m/s are assumed to be GPS multipathing or
    other recording errors and are removed. Deviation change indicates an
    unexpected delay. Stop delay is true if the nextstop attribute changed from
    the previous location.

    Args:
        daily_results: A Pandas Dataframe object containing bus location, time,
            and other RDS data.

    Returns:
        A Pandas Dataframe object containing the cleaned set of results with an
        additional columns for calculated variables.
    """
    # Remove duplicate trip locations
    daily_results.drop_duplicates(subset=['tripid', 'locationtime'], inplace=True)
    daily_results.sort_values(by=['tripid', 'locationtime'], inplace=True)

    # Offset tripdistance, locationtime, and tripids by 1
    daily_results['prev_tripdistance'] = 1
    daily_results['prev_locationtime'] = 1
    daily_results['prev_deviation'] = 1
    daily_results['prev_tripid'] = 1
    daily_results['prev_stopid'] = 1
    daily_results['prev_tripdistance'] = daily_results['tripdistance'].shift(1)
    daily_results['prev_locationtime'] = daily_results['locationtime'].shift(1)
    daily_results['prev_deviation'] = daily_results['scheduledeviation'].shift(1)
    daily_results['prev_tripid'] = daily_results['tripid'].shift(1)
    daily_results['prev_stopid'] = daily_results['nextstop'].shift(1)

    # Remove NA rows, and rows where tripid is different (last recorded location)
    daily_results.loc[daily_results.tripid != daily_results.prev_tripid, 'tripid'] = None
    daily_results.dropna(inplace=True)

    # If no rows are left, return empty dataframe
    if daily_results.size == 0:
        return daily_results

    # Calculate average speed between each location bus is tracked at
    daily_results.loc[:, 'dist_diff'] = daily_results['tripdistance'] \
        - daily_results['prev_tripdistance']
    daily_results.loc[:, 'time_diff'] = daily_results['locationtime'] \
        - daily_results['prev_locationtime']
    daily_results.loc[:, 'speed_m_s'] = daily_results['dist_diff'] \
        / daily_results['time_diff']

    # Calculate change in schedule deviation
    daily_results.loc[:, 'deviation_change_s'] = daily_results['scheduledeviation'] \
        - daily_results['prev_deviation']

    # Find rows where the delay/speed incorporated a transit stop (nextstop changed)
    daily_results.loc[daily_results['nextstop'] != daily_results['prev_stopid'], 'at_stop'] = True
    daily_results.loc[daily_results['nextstop'] == daily_results['prev_stopid'], 'at_stop'] = False

    # Remove rows where speed is below 0 or above 30 and round
    daily_results = daily_results[daily_results['speed_m_s'] >= 0]
    daily_results = daily_results[daily_results['speed_m_s'] <= 30]
    daily_results.loc[:, 'speed_m_s'] = round(
        daily_results.loc[:, 'speed_m_s'])

    # Remove rows where schedule deviation change is below -300 or above 300 (5mins)
    daily_results = daily_results[daily_results['deviation_change_s'] >= -300]
    daily_results = daily_results[daily_results['deviation_change_s'] <= 300]
    daily_results = daily_results[daily_results['deviation_change_s'] != 0]
    daily_results.loc[:, 'deviation_change_s'] = round(
        daily_results.loc[:, 'deviation_change_s'])

    # Merge scraped data with the gtfs data to get route ids
    gtfs_trips = pd.read_csv('./transit_vis/data/google_transit/trips.txt')
    gtfs_trips = gtfs_trips[['route_id', 'trip_id', 'trip_short_name']]
    gtfs_routes = pd.read_csv('./transit_vis/data/google_transit/routes.txt')
    gtfs_routes = gtfs_routes[['route_id', 'route_short_name']]
    daily_results = daily_results.merge(
        gtfs_trips,
        left_on='tripid',
        right_on='trip_id')
    daily_results = daily_results.merge(
        gtfs_routes,
        left_on='route_id',
        right_on='route_id')
    return daily_results

def get_nearest(src_points, candidates):
    """Find nearest neighbors for all source points from a set of candidates.

    Taken from:
    https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
    Uses a BallTree implementation from sklearn to find closest points between a
    list of source and candidate points in an efficient manner. Uses Haversine
    (great sphere) distance, which means points should be in lat/lon coordinate
    format.

    Args:
        src_points: A Pandas Dataframe consisting of 2 columns named 'lat' and
            'lon'. Each row is a bus coordinate to be assigned.
        candidates: A Pandas Dataframe consisting of 2 columns named 'lat' and
            'lon'. Each row is a point within a segment on a bus route.

    Returns:
        A tuple with 2 lists; the first contains the index of the closest
        candidate point to each source point, the second contains the distance
        between those points.
    """
    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=1)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest_idx = indices[0]
    closest_dist = distances[0]
    return closest_idx

def assign_results_to_segments(kcm_routes, daily_results):
    """Assigns each of the bus locations from the RDS to the closest segment.

    Assigns each location that a bus was tracked at to its closest segment by
    using the get_nearest function to calculate distances between bus locations
    and points on segments in the routes geojson. A bus location can only be
    assigned to a line segment from the route that it belongs to.

    Args:
        kcm_routes: A geojson file (generated during initialize_dynamodb.py)
            that contains features for each segment in the bus network.
        daily_results: A Pandas Dataframe containing the preprocessed data
            queried from the RDS data warehouse.

    Returns:
        A Pandas Dataframe containing the bus location data passed with
        additional columns for the closest route and segment ids.
    """
    # Convert segment data from json format to tabular
    # vis_id is unique id, route_id helps narrow down when matching segments
    feature_coords = []
    feature_lengths = []
    compkeys = []
    route_ids = []
    for feature in kcm_routes['features']:
        assert feature['geometry']['type'] == 'MultiLineString'
        for coord_pair in feature['geometry']['coordinates'][0]:
            feature_coords.append(coord_pair)
            feature_lengths.append(feature['properties']['seg_length'])
            compkeys.append(feature['properties']['COMPKEY'])
            route_ids.append(feature['properties']['ROUTE_ID'])
    segments = pd.DataFrame()
    segments['route_id'] = route_ids
    segments['compkey'] = compkeys
    segments['length'] = feature_lengths
    segments['lat'] = np.array(feature_coords)[:,1] # Not sure why, but these are lon, lat in the geojson
    segments['lon'] = np.array(feature_coords)[:,0] # Not sure why, but these are lon, lat in the geojson

    # Find closest segment that shares route for each tracked location
    to_upload = pd.DataFrame()
    route_list = pd.unique(daily_results['route_id'])
    untracked = []
    for route in route_list:
        route_results = daily_results.loc[daily_results['route_id']==route].copy()
        route_segments = segments.loc[segments['route_id']==route].reset_index()
        if len(route_results) > 0 and len(route_segments) > 0:
            result_idxs = get_nearest(route_results[['lat', 'lon']], route_segments[['lat', 'lon']])
            result_segs = route_segments.iloc[result_idxs, :]
            route_results['seg_lat'] = np.array(result_segs['lat'])
            route_results['seg_lon'] = np.array(result_segs['lon'])
            route_results['seg_length'] = np.array(result_segs['length'])
            route_results['seg_compkey'] = np.array(result_segs['compkey'])
            route_results['seg_route_id'] = np.array(result_segs['route_id'])
            to_upload = to_upload.append(route_results)
        else:
            untracked.append(route)
            result_idxs = -1
            result_dists = -1
    print(f"Routes {untracked} are either not tracked, or do not have an id in the KCM shapefile")
    # Clean up the results columns
    columns_to_keep = [
        # From the database and its offsets
        'tripid',
        'vehicleid',
        'lat',
        'lon',
        'orientation',
        'scheduledeviation',
        'prev_deviation',
        'totaltripdistance',
        'tripdistance',
        'prev_tripdistance',
        'closeststop',
        'nextstop',
        'prev_stopid',
        'locationtime',
        'prev_locationtime',
        # Calculated from database values
        'dist_diff',
        'time_diff',
        'speed_m_s',
        'deviation_change_s',
        'at_stop',
        # From joining GTFS
        'route_id',
        'trip_short_name',
        'route_short_name',
        # From joining nearest kcm segments
        'seg_compkey',
        'seg_length', # Should be in meters
        'seg_route_id',
        'seg_lat',
        'seg_lon']
    to_upload = to_upload[columns_to_keep]
    return to_upload

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'pct_%s' % n
    return percentile_

def summarize_rds(rds_limit, split_data, update_gtfs, save_locally, save_dates, table_name, unique_trip_col, outdir=None):
    """Queries 24hrs of data from RDS, calculates speeds, and uploads them.

    Runs daily to take 24hrs worth of data stored in the data warehouse
    and summarize it for usage with the Folium map. Speeds for each observed
    bus location are calculated using consecutive trip distances and times.The
    geojson segments used are generated during initialize_dynamodb.py, which
    guarantees that they will be the same ones that are stored on the dynamodb
    database, allowing for this script to upload them. The Folium map will then
    download the speeds and display them using the same geojson file once again.

    Args:
        geojson_name: Path to the geojson file that is to be uploaded. Do not
            include file type ending (.geojson etc.).
        dynamodb_table_name: The name of the table containing the segments that
            speeds will be matched and uploaded to.
        rds_limit: An integer specifying the maximum number of rows to query.
            Useful for debugging and checking output before making larger
            queries. Set to 0 for no limit.
        split_data: An integer specifying how many blocks to split each 24hr query
            into.
        save_locally: Boolean specifying whether to save the processed data to
            a folder on the user's machine.
        save_dates: List of strings containing days that data should be queried for.
        upload: Boolean specifying whether to upload the processed data to the
            dynamodb table.

    Returns:
        An integer of the number of segments that were updated in the
        database.
    """
    # Update the current gtfs trip-route info from King County Metro
    if update_gtfs:
        print("Updating the GTFS files...")
        update_gtfs_route_info()

    # Load scraped data, return if there is no data found, process otherwise
    print("Connecting to RDS...")
    conn = connect_to_rds()

    for day in save_dates:
        end_time = round(datetime.strptime(day, '%Y-%m-%d').timestamp()) + (24*60*60)
        print(f"Querying {day} data from RDS (~5mins if no limit specified)...")
        all_daily_results = []
        # Break up the query into {split_data} pieces
        for i in range(0, split_data):
            start_time = int(round(end_time - (24*60*60/split_data), 0))
            daily_results = get_results_by_time(conn, start_time, end_time, rds_limit, table_name, unique_trip_col)
            if daily_results is None:
                print(f"No results found for {start_time}")
                end_time = start_time
                continue
            print(f"Processing queried RDS data...{i+1}/{split_data}")
            # daily_results = preprocess_trip_data(daily_results)
            all_daily_results.append(daily_results)
            del daily_results
            end_time = start_time
        if len(all_daily_results) == 0:
            continue
        daily_results = pd.concat(all_daily_results)

        # If no data collected; return empty
        if len(daily_results) == 0:
            print("No data found.")
            continue

        # Save the processed data for the user if specified
        if save_locally:
            outfile = datetime.utcfromtimestamp(start_time).replace(tzinfo=pytz.utc).astimezone(pytz.timezone(os.getenv('TZ'))).strftime('%Y_%m_%d')
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            print("Saving processed speeds to data folder...")
            with open(f"{outdir}/{outfile}.pkl", 'wb') as f:
                pickle.dump(daily_results, f)

        print(f"Date: {day} Number of tracks: {len(daily_results)}")

    return len(daily_results)
