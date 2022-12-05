from database import summarize_rds
from database import data_utils
from datetime import datetime, timedelta
from dotenv import load_dotenv

import os


if __name__ == "__main__":
    load_dotenv()
    # Collect data from days where validation was recorded
    # date_list = data_utils.get_validation_dates("./data/kcm_validation_sensor")

    # Collect from current-start_back_days-numdays -> current-start_back_days
    date_list = []
    start_back_days = 1
    num_days = 366
    current_day = datetime.now()
    start_day = current_day - timedelta(days=start_back_days)
    for x in range (0, num_days):
        date_list.append((start_day - timedelta(days=x)).strftime('%Y-%m-%d'))
    
    # Print dates data is to be downloaded for
    print(date_list)

    print(os.getenv('PWD'))

    # Download the data
    NUM_TRACKS = summarize_rds.summarize_rds(
        dynamodb_table_name='KCM_Bus_Routes',
        rds_limit=0,
        split_data=3,
        update_gtfs=True,
        save_locally=True,
        save_dates=date_list,
        upload=False,
        outdir='./data/kcm_all')
    print(f"Number of tracks for last day: {NUM_TRACKS}")
