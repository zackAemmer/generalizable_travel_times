import numpy as np
import torch

from utils import data_utils, data_loader


class PersistentTimeSeqModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.requires_grid = False
        self.collate_fn = data_loader.sequential_collate
        self.train_time = 0.0
        self.hyperparameter_dict = {'BATCH_SIZE': 512}
        self.is_nn = False
        return None
    def train(self, dataloader, config):
        return None
    def evaluate(self, dataloader, config):
        data = [x for x in dataloader]
        seq_lens = np.concatenate([x[0][2] for x in data])
        preds = np.array([x*30 for x in seq_lens])
        all_labels = []
        for x in data:
            labels = data_utils.de_normalize(x[1].numpy(), config['time_calc_s_mean'], config['time_calc_s_std'])
            mask = data_utils.create_tensor_mask(x[0][2]).numpy()
            labels = data_utils.aggregate_tts(labels, mask)
            all_labels.append(labels)
        labels = np.concatenate(all_labels)
        return labels, preds
    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None