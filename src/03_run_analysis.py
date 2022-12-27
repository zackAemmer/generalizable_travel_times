from database import data_utils
from dotenv import load_dotenv


if __name__ == "__main__":
    with open('10M.pkl', 'rb') as f:
        df = pickle.load(f)