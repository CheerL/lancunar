import os
import sys

import numpy as np

import DataManagerNii as DMNII
from model import Model

if __name__ == '__main__':
    basePath = os.getcwd() # get current path
    params = dict() # all parameters
    params['DataManagerParams'] = dict() # parameters for data manager class
    params['ModelParams'] = dict() # parameters for model
    params['TestParams'] = dict() # parameters for testing
    # params of the algorithm
    params['ModelParams']['device'] = 0 # the id of the GPU
    params['ModelParams']['snapshot'] = 0 #85000
    params['ModelParams']['dirTrain'] = 'data/Lancunar/Lacunar_training/' # the directory of training data
    params['ModelParams']['dirValidation'] = 'data/Lancunar/Lacunar_validation/' # the directory of training data
    params['ModelParams']['dirTest']='data/Lancunar/Lacunar_testing/' #the directory of the testing data
    # params['ModelParams']['dirTrain'] = '../CMB/mini_training/' # the directory of training data
    # params['ModelParams']['dirValidation']='../CMB/mini_validation/' #the directory of the validation data
    # params['ModelParams']['dirTest']='../CMB/mini_testing/' #the directory of the testing data
    params['ModelParams']['dirResult'] = "result/" # the directory of the results of testing data
    # where to save the models while training
    params['ModelParams']['dirLog'] = "log/"
    params['ModelParams']['dirSnapshots'] = "snapshot/" # the directory of the model snapshots for training
    params['ModelParams']['tailSnapshots'] = 'WL/mini_vnet/' # the full path of the model snapshots is the join of dirsnapshots and presnapshots
    params['ModelParams']['batchsize'] = 100  # the batch size
    params['ModelParams']['iteration'] = 1000000
    params['ModelParams']['baseLR'] = 1e-4  # the learning rate, initial one
    params['ModelParams']['weight_decay'] = 0.0005

    params['ModelParams']['valInterval'] = 500  # the number of training interations between testing
    params['ModelParams']['trainInterval'] = 20  # the number of training interations between testing
    params['ModelParams']['loss'] = 'nll'
    # params of the DataManager
    params['DataManagerParams']['epoch'] = 5000  # the number of total training iterations
    params['DataManagerParams']['feedThreadNum'] = 8  # the number of threads to do data augmentation
    params['DataManagerParams']['loadThreadNum'] = 64
    params['DataManagerParams']['VolSize'] = np.asarray([64, 64, 4], dtype=int) # the size of the crop image
    params['DataManagerParams']['TestStride'] = np.asarray([64, 64, 4], dtype=int) # the stride of the adjacent crop image in testing phase and validation phase
    params['DataManagerParams']['TrainStride'] = np.asarray([32, 32, 2], dtype=int) # the stride of the adjacent crop image in testing phase and validation phase
    params['DataManagerParams']['MaxEmpty'] = 1
    params['DataManagerParams']['dataQueueSize'] = 1000
    params['DataManagerParams']['posQueueSize'] = 4000

    # Ture: produce the probaility map in the testing phase, False: produce the  label image
    params['TestParams']['ProbabilityMap'] = False
    train = [i for i, j in enumerate(sys.argv) if j == '-train']
    if len(train) > 0:
        dataManagerTrain = DMNII.DataManagerNii(
            params['ModelParams']['dirTrain'],
            params['ModelParams']['dirResult'],
            params['DataManagerParams']
            )
        dataManagerTrain.load_data()  # loads in sitk format
        dataManagerTrain.run_feed_thread()
        model = Model(params)
        model.train(dataManagerTrain) #train model

    # test = [i for i, j in enumerate(sys.argv) if j == '-test']
    # for i in sys.argv:
    #     if i.isdigit():
    #         snapnumber = i
    #         break

    # if len(test) > 0:
    #     model.test(snapnumber) # test model, the snapnumber is the number of the model snapshot
