#!/usr/bin python3


import gc
import itertools
import json
import os
import random
import time

import numpy as np
import torch
from sklearn import metrics
from tabulate import tabulate

from models import avg_speed, conv, ff, persistent, rnn, schedule, transformer
from utils import data_loader, data_utils, model_utils

from torch.profiler import profile, record_function, ProfilerActivity

import os.path as path

def run():
    full_set = 2300000*5
    data = np.ones(full_set*12*64,dtype="float32").reshape(full_set,12,64)
    data_utils.write_pkl(data)
    filename = path.join('newfile.dat')
    fp = np.memmap(filename, dtype='object', mode='w+', shape=(100000,12))
    fp[:] = data[:]
    fp.flush()
    newfp = np.memmap(filename, dtype='object', mode='r', shape=data.shape)

    return None

if __name__=="__main__":
    run()