import numpy as np

from utils import data_utils


class PersistentSpeedSeqModel:
    def __init__(self, config, train_mask):
        self.config = config
        self.train_mask = train_mask
        return None

    def predict(self, dataloader, test_mask):
        data, labels = dataloader.dataset.tensors
        # Predict that next speed will be same as present speed
        speeds = np.array([x[0][0,6].numpy() for x in data], dtype='float32')
        preds = data_utils.de_normalize(speeds, self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        # Compare to avg trajectory speed
        labels = np.array([np.mean(x.numpy()[test_mask[i]]) for i,x in enumerate(labels)])
        labels = data_utils.de_normalize(labels, self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        return labels, preds

    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None