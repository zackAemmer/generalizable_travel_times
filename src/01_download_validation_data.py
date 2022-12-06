from database import summarize_rds
from database import data_utils
from datetime import datetime, timedelta
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    # Collect data from days where validation was recorded
    # date_list = data_utils.get_validation_dates("./data/kcm_validation_sensor")

    # Collect from current-start_back_days-num_days -> current-start_back_days
    # Norway collection began: 2022/2/10
    # KCM collection began: 2020/9/24
    date_list = []
    start_back_days = 307
    num_days = 600
    current_day = datetime.now()
    start_day = current_day - timedelta(days=start_back_days)
    for x in range (0, num_days):
        date_list.append((start_day - timedelta(days=x)).strftime('%Y-%m-%d'))

    # Print dates data is to be downloaded for
    print(f"Date start: {date_list[0]}, date end: {date_list[-1]}")

    # Download the data
    NUM_TRACKS = summarize_rds.summarize_rds(
        rds_limit=0,
        split_data=3,
        update_gtfs=False,
        save_locally=True,
        save_dates=date_list,
        table_name='active_trips_study',
        unique_trip_col='tripid',
        outdir='./data/kcm_all')
    print(f"Number of tracks for last day: {NUM_TRACKS}")
