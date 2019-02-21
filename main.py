import os
import sys

import numpy as np

import data_manager
from model import Model
from config import model_params

if __name__ == '__main__':
    basePath = os.getcwd() # get current path
    model = Model(model_params)
    if '-train' in sys.argv:
        model.train() #train model

    if '-test' in sys.argv:
        model.test() # test model, the snapnumber is the number of the model snapshot

    if '-lr' in sys.argv:
        model.find_lr(1e-4, 10, 500, 0.8)
