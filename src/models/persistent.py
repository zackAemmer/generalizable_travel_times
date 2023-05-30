import numpy as np

from utils import data_utils


# class PersistentSpeedSeqModel:
#     def __init__(self, model_name, config, min_speed):
#         self.model_name = model_name
#         self.config = config
#         self.min_speed = min_speed
#         self.train_time = 0.0
#         return None
#     def train(self, dataloader, config):
#         return None
#     def evaluate(self, dataloader, config):
#         context, X, y, seq_lens = data_utils.extract_all_dataloader(dataloader, sequential_flag=True)
#         max_len = max(seq_lens)
#         # Carry the starting speeds through full sequences
#         preds = X[:,0,8].unsqueeze(1).repeat((1,max_len)).numpy()
#         preds = data_utils.de_normalize(preds, config['speed_m_s_mean'], config['speed_m_s_std'])
#         # Assume a minimum speed, otherwise infinite/exploding travel times are possible (0 m/s)
#         preds[preds<self.min_speed] = self.min_speed
#         labels = y.numpy()
#         labels = data_utils.de_normalize(labels, config['speed_m_s_mean'], config['speed_m_s_std'])
#         return labels, preds
#     def save_to(self, path):
#         data_utils.write_pkl(self, path)
#         return None

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