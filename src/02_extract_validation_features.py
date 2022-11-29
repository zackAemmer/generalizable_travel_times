from database import data_utils
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    # Extract features from downloaded validation data
    NUM_SEGMENTS_UPDATED = data_utils.extract_validation_trips("./data/kcm_validation_sensor_tracks")
    print(f"Number of tracks for last day: {NUM_SEGMENTS_UPDATED}")