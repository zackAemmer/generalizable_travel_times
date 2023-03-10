"""
Download all days of data on S3 that are not currently in the given data directory.
"""

import boto3
import os

from dotenv import load_dotenv


def download_new_s3_files(data_folder, bucket_name):
    print(f"Getting new files for {data_folder} from S3 bucket {bucket_name}")
    downloaded_files = os.listdir(data_folder)
    print(f"Found {len(downloaded_files)} downloaded files")
    try:
        cli = boto3.client(
            's3',
            aws_access_key_id=os.getenv('ACCESS_KEY'),
            aws_secret_access_key=os.getenv('SECRET_KEY')
        )
        print(f"Successfully connected to S3")
        # See what is on S3
        response = cli.list_objects_v2(Bucket=bucket_name)
        available_files = [x['Key'] for x in response['Contents']]
        # Get list of files that are not already downloaded
        new_files = [x for x in available_files if x not in downloaded_files]
        print(f"Found {len(new_files)} new files to download out of {len(available_files)} files in the specified bucket")
        # Download all new files to same data folder
        for i,file in enumerate(new_files):
            print(f"Downloading file {i} out of {len(new_files)}")
            cli.download_file(bucket_name, file, f"{data_folder}{file}")
    except ValueError:
        print(f"Failure to access S3")
    return None

if __name__ == "__main__":
    load_dotenv()
    download_new_s3_files("./data/kcm_all/", "gtfs-collection-kcm")
    download_new_s3_files("./data/nwy_all/", "gtfs-collection-nwy")