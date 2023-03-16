import numpy as np

from utils import data_utils


class PersistentSpeedSeqModel:
    def __init__(self, config, min_value):
        self.config = config
        self.min_value = min_value
        return None

    def predict(self, dataloader, min_speed=2.0):
        context, X, y, seq_lens = data_utils.extract_all_dataloader(dataloader, sequential_flag=True)
        max_len = max(seq_lens)
        # Carry the starting speeds through full sequences
        preds = X[:,0,5].unsqueeze(1).repeat((1,max_len)).numpy()
        preds = data_utils.de_normalize(preds, self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        # Assume a minimum speed, otherwise infinite travel times are possible (0 m/s)
        preds[preds<min_speed] = min_speed
        labels = y.numpy()
        labels = data_utils.de_normalize(labels, self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        return labels, preds
        # data, labels = dataloader.dataset.tensors
        # # Predict that next speed will be same as present speed
        # speeds = np.array([x[0][0,6].numpy() for x in data], dtype='float32')
        # speeds = data_utils.de_normalize(speeds, self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        # # Any speeds at (or close to) 0 m/s should be replaced, since they give inf tt
        # speeds[speeds<np.mean(speeds)] = self.min_value
        # # Repeat and pad predictions to get values for every step of sequence
        # max_len = data[0][0].shape[0]
        # preds = [np.repeat(x,np.sum(test_mask[i])) for i,x in enumerate(speeds)]
        # preds = np.array([np.pad(x,(0,max_len-len(x))) for x in preds], dtype='float32')
        # # Compare to avg trajectory speed
        # # Must not modify dataloader
        # norm_labels = labels.numpy().copy()
        # norm_labels[test_mask] = data_utils.de_normalize(norm_labels[test_mask], self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        # return norm_labels, preds

    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None