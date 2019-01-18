import os
import sys

import numpy as np

import data_manager
from model import Model

if __name__ == '__main__':
    basePath = os.getcwd() # get current path
    params = dict() # all parameters
    data_manager_params = dict() # parameters for data manager class
    model_params = dict() # parameters for model
    test_params = dict() # parameters for testing
    # params of the algorithm
    model_params['device'] = 0 # the id of the GPU
    model_params['snapshot'] = 5000 #85000
    model_params['dirTrain'] = 'data/Lancunar/Lacunar_training/' # the directory of training data
    model_params['dirValidation'] = 'data/Lancunar/Lacunar_validation/' # the directory of training data
    model_params['dirTest']='data/Lancunar/Lacunar_testing/' #the directory of the testing data
    # model_params['dirTrain'] = '../CMB/mini_training/' # the directory of training data
    # model_params['dirValidation']='../CMB/mini_validation/' #the directory of the validation data
    # model_params['dirTest']='../CMB/mini_testing/' #the directory of the testing data
    model_params['dirResult'] = "result/" # the directory of the results of testing data
    # where to save the models while training
    model_params['dirLog'] = "log/"
    model_params['dirSnapshots'] = "snapshot/" # the directory of the model snapshots for training
    model_params['tailSnapshots'] = 'WL/mini_vnet/' # the full path of the model snapshots is the join of dirsnapshots and presnapshots
    model_params['batchsize'] = 150  # the batch size
    model_params['iteration'] = 1000000
    model_params['baseLR'] = 1e-4  # the learning rate, initial one
    model_params['weight_decay'] = 0.0005

    model_params['valInterval'] = 500  # the number of training interations between testing
    model_params['trainInterval'] = 50  # the number of training interations between testing
    model_params['loss'] = 'nll'
    # params of the DataManager
    data_manager_params['feedThreadNum'] = 8  # the number of threads to do data augmentation
    data_manager_params['loadThreadNum'] = 64
    data_manager_params['VolSize'] = (64, 64, 4) # the size of the crop image
    data_manager_params['batchsize'] = 100  # the batch size
    data_manager_params['MaxEmpty'] = 1
    data_manager_params['dataQueueSize'] = 20

    model_params['dataManager'] = data_manager_params

    # Ture: produce the probaility map in the testing phase, False: produce the  label image
    # params['TestParams']['ProbabilityMap'] = False
    model = Model(model_params)
    if '-train' in sys.argv:
        model.train() #train model

    if '-test' in sys.argv:
        model.test() # test model, the snapnumber is the number of the model snapshot
