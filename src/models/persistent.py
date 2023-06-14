import numpy as np

from utils import data_utils


class PersistentTimeSeqModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.train_time = 0.0
        return None
    def train(self, dataloader, config):
        return None
    def evaluate(self, dataloader, config):
        context, X, y, seq_lens = data_utils.extract_all_dataloader(dataloader, sequential_flag=True)
        max_len = max(seq_lens)
        # Predict 30 seconds for all sequence points
        preds = np.ones((y.shape)) * 30.0
        labels = y.numpy()
        labels = data_utils.de_normalize(labels, config['time_gap_mean'], config['time_gap_std'])
        _, mask = data_utils.get_seq_info(dataloader)
        preds = data_utils.aggregate_tts(preds, mask)
        labels = data_utils.aggregate_tts(labels, mask)
        return labels, preds
    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None