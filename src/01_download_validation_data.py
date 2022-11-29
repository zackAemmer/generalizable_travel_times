from database import summarize_rds
from database import data_utils
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    # Collect data from days where validation was recorded
    date_list = data_utils.get_validation_dates("./data/kcm_validation_sensor_tracks")

    NUM_SEGMENTS_UPDATED = summarize_rds.summarize_rds(
        geojson_name='./transit_vis/data/streets_routes_0001buffer',
        dynamodb_table_name='KCM_Bus_Routes',
        rds_limit=0,
        split_data=3,
        update_gtfs=True,
        save_locally=True,
        save_dates=date_list,
        upload=False)
    print(f"Number of tracks for last day: {NUM_SEGMENTS_UPDATED}")