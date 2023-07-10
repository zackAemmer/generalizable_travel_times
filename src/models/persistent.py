import numpy as np

from utils import data_utils, data_loader


class PersistentTimeSeqModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.requires_grid = False
        self.collate_fn = data_loader.sequential_collate
        self.train_time = 0.0
        self.is_nn = False
        return None
    def train(self, dataloader, config):
        return None
    def evaluate(self, dataloader, config):
        data = np.array(dataloader.dataset.content)[dataloader.sampler.indices]
        preds = np.array([len(sample['time_calc_s'])*30 for sample in data])
        labels = np.array([sample['time'] for sample in data])
        return labels, preds
    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None