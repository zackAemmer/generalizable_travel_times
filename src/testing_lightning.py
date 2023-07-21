#!/usr/bin python3


import numpy as np
import h5py

import os
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger

from models import ff
from utils import new_data_loader
import json

def run(run_folder, train_network_folder, **kwargs):
    embed_dict = {
        'timeID': {
            'vocab_size': 1440,
            'embed_dims': 27
        },
        'weekID': {
            'vocab_size': 7,
            'embed_dims': 4
        }
    }
    hyperparameter_dict = {
        'FF': {
            'EPOCHS': 2,
            'BATCH_SIZE': 512,
            'LEARN_RATE': .001,
            'HIDDEN_SIZE': 32,
            'NUM_LAYERS': 2,
            'DROPOUT_RATE': .1
        },
        'CONV': {
            'EPOCHS': 2,
            'BATCH_SIZE': 512,
            'LEARN_RATE': .001,
            'HIDDEN_SIZE': 32,
            'NUM_LAYERS': 2,
            'DROPOUT_RATE': .1
        },
        'GRU': {
            'EPOCHS': 2,
            'BATCH_SIZE': 512,
            'LEARN_RATE': .001,
            'HIDDEN_SIZE': 32,
            'NUM_LAYERS': 2,
            'DROPOUT_RATE': .1
        },
        'TRSF': {
            'EPOCHS': 2,
            'BATCH_SIZE': 512,
            'LEARN_RATE': .001,
            'HIDDEN_SIZE': 32,
            'NUM_LAYERS': 2,
            'DROPOUT_RATE': .1
        },
        'DEEPTTE': {
            'EPOCHS': 2,
            'BATCH_SIZE': 512,
            'LEARN_RATE': .001
        }
    }
    
    with open(f"{run_folder}{train_network_folder}deeptte_formatted/train_summary_config.json", "r") as f:
        config = json.load(f)
    dataset = new_data_loader.LoadSliceDataset(f"{run_folder}{train_network_folder}deeptte_formatted/train", config)
    ffmodel = ff.FF_L(model_name="FF", n_features=12, collate_fn=new_data_loader.basic_collate, hyperparameter_dict=hyperparameter_dict['FF'], embed_dict=embed_dict)
    loader = DataLoader(dataset, collate_fn=ffmodel.collate_fn, batch_size=512)

    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, logger=CSVLogger(save_dir="logs/"))
    trainer.fit(model=ffmodel, train_dataloaders=loader)

    # # define any number of nn.Modules (or use your current ones)
    # encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    # decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    # # define the LightningModule
    # class LitAutoEncoder(pl.LightningModule):
    #     def __init__(self, encoder, decoder):
    #         super().__init__()
    #         self.encoder = encoder
    #         self.decoder = decoder
    #     def training_step(self, batch, batch_idx):
    #         # training_step defines the train loop.
    #         # it is independent of forward
    #         x, y = batch
    #         x = x.view(x.size(0), -1)
    #         z = self.encoder(x)
    #         x_hat = self.decoder(z)
    #         loss = nn.functional.mse_loss(x_hat, x)
    #         # Logging to TensorBoard (if installed) by default
    #         self.log("train_loss", loss)
    #         return loss
    #     def configure_optimizers(self):
    #         optimizer = optim.Adam(self.parameters(), lr=1e-3)
    #         return optimizer

    # # init the autoencoder
    # autoencoder = LitAutoEncoder(encoder, decoder)

    # # setup data
    # dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    # train_loader = DataLoader(dataset, num_workers=4)

    # # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    # trainer = pl.Trainer(limit_train_batches=100, max_epochs=1, logger=CSVLogger(save_dir="logs/"))
    # trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    # # load checkpoint
    # checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    # autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

    # # choose your trained nn.Module
    # encoder = autoencoder.encoder
    # encoder.eval()

    # # embed 4 fake images!
    # fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
    # embeddings = encoder(fake_image_batch)
    # print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

if __name__=="__main__":
    run(
        run_folder="./results/debug/",
        train_network_folder="kcm/",
        test_network_folder="atb/",
        tune_network_folder="atb/",
        TUNE_EPOCHS=2,
        BATCH_SIZE=32,
        LEARN_RATE=1e-3,
        HIDDEN_SIZE=32,
        grid_s_size=500,
        data_subset=.1,
        n_tune_samples=100,
        n_folds=2,
        holdout_routes=[100252,100139,102581,100341,102720]
    )