import numpy as np

from utils import data_utils


class PersistentSpeedSeqModel:
    def __init__(self, config, train_mask, min_value):
        self.min_value = min_value
        self.config = config
        self.train_mask = train_mask
        return None

    def predict(self, dataloader, test_mask):
        data, labels = dataloader.dataset.tensors
        # Predict that next speed will be same as present speed
        speeds = np.array([x[0][0,6].numpy() for x in data], dtype='float32')
        speeds = data_utils.de_normalize(speeds, self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        # Any speeds at (or close to) 0 m/s should be replaced, since they give inf tt
        speeds[speeds<np.mean(speeds)] = self.min_value
        # Repeat and pad predictions to get values for every step of sequence
        max_len = data[0][0].shape[0]
        preds = [np.repeat(x,np.sum(test_mask[i])) for i,x in enumerate(speeds)]
        preds = np.array([np.pad(x,(0,max_len-len(x))) for x in preds], dtype='float32')
        # Compare to avg trajectory speed
        # Must not modify dataloader
        norm_labels = labels.numpy().copy()
        norm_labels[test_mask] = data_utils.de_normalize(norm_labels[test_mask], self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        return norm_labels, preds

    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None