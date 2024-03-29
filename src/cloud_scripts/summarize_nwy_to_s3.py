#!/usr/bin python3
import boto3
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle
import secret


def get_time_info(time_delta=0):
    # Get the UTC
    utc = datetime.utcnow()
    adj = timedelta(hours=time_delta)
    target_time = (utc + adj)
    date_str = target_time.strftime("%Y_%m_%d_%H")
    epoch = round(utc.timestamp())
    return date_str, epoch


if __name__ == "__main__":
    # Load data
    scrape_folder = "./scraped_data/nwy/"
    scrape_files = os.listdir(scrape_folder)
    all_data = []
    date_str, current_epoch = get_time_info(1)
    for filename in scrape_files:
        if filename[-4:]==".pkl" and filename[:10]!=date_str[:10]:
            with open(scrape_folder+filename, 'rb') as f:
                data = pickle.load(f)
            all_data.append(data)
            os.remove(scrape_folder+filename)
    # Combine and remove duplicated locations
    try:
        all_data = pd.concat(all_data)
        all_data = all_data.drop_duplicates(['trip_id','locationtime']).sort_values(['trip_id','locationtime'])
        # Upload to S3
        date_str, current_epoch = get_time_info(1)
        cli = boto3.client(
            's3',
            aws_access_key_id=secret.ACCESS_KEY,
            aws_secret_access_key=secret.SECRET_KEY
        )
        cli.put_object(
            Body=pickle.dumps(all_data),
            Bucket="gtfs-collection-nwy",
            Key=date_str[:10]+".pkl"
        )
    except:
        print(f"Either no files found for {date_str}, or failure to access S3")
