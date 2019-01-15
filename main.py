import sys
import os
import numpy as np
import model as model
import DataManagerNii as DMNII
from multiprocessing import Process, Queue


if __name__ == '__main__':
    basePath = os.getcwd() # get current path
    params = dict() # all parameters
    params['DataManagerParams'] = dict() # parameters for data manager class
    params['ModelParams'] = dict() # parameters for model
    params['TestParams'] = dict() # parameters for testing
    # params of the algorithm
    params['ModelParams']['device'] = 0 # the id of the GPU
    params['ModelParams']['snapshot'] = 0 #85000
    # params['ModelParams']['dirTrain'] = 'data/Lancunar/mini_train/' # the directory of training data
    # params['ModelParams']['dirTest']='data/Lancunar/Lacunar_testing/' #the directory of the testing data
    # params['ModelParams']['dirResult']="/home/ftp/data/output/" #where we need to save the results (relative to the base path)
    params['ModelParams']['dirTrain'] = '../CMB/mini_training/' # the directory of training data
    params['ModelParams']['dirValidation']='../CMB/mini_validation/' #the directory of the validation data
    params['ModelParams']['dirTest']='../CMB/mini_testing/' #the directory of the testing data
    params['ModelParams']['dirResult'] = "result/" # the directory of the results of testing data
    # where to save the models while training
    params['ModelParams']['dirLog'] = "log/"
    params['ModelParams']['dirSnapshots'] = "snapshot/" # the directory of the model snapshots for training
    params['ModelParams']['tailSnapshots'] = 'WL/mini_vnet/' # the full path of the model snapshots is the join of dirsnapshots and presnapshots
    params['ModelParams']['batchsize'] = 1  # the batch size
    params['ModelParams']['iteration'] = 1000000
    params['ModelParams']['baseLR'] = 5e-4  # the learning rate, initial one
    params['ModelParams']['weight_decay'] = 0.0005

    params['ModelParams']['valInterval'] = 500  # the number of training interations between testing
    params['ModelParams']['trainInterval'] = 20  # the number of training interations between testing

    # params of the DataManager
    params['DataManagerParams']['epoch'] = 5000  # the number of total training iterations
    params['DataManagerParams']['nProc'] = 4  # the number of threads to do data augmentation
    params['DataManagerParams']['VolSize'] = np.asarray([64, 64, 16], dtype=int) # the size of the crop image
    params['DataManagerParams']['TestStride'] = np.asarray([64, 64, 16], dtype=int) # the stride of the adjacent crop image in testing phase and validation phase
    params['DataManagerParams']['TrainStride'] = np.asarray([32, 32, 8], dtype=int) # the stride of the adjacent crop image in testing phase and validation phase
    params['DataManagerParams']['MaxEmpty'] = 2
    params['DataManagerParams']['dataQueueSize'] = 200
    params['DataManagerParams']['posQueueSize'] = 1000

    # Ture: produce the probaility map in the testing phase, False: produce the  label image
    params['TestParams']['ProbabilityMap'] = False
    model = model.Model(params)
    train = [i for i, j in enumerate(sys.argv) if j == '-train']
    if len(train) > 0:
        dataManagerTrain = DMNII.DataManagerNii(
            params['ModelParams']['dirTrain'],
            params['ModelParams']['dirResult'],
            params['DataManagerParams']
            )
        dataManagerTrain.load_data()  # loads in sitk format
        dataManagerTrain.run_train_processes()
        # thread creation
        model.train(dataManagerTrain) #train model

    test = [i for i, j in enumerate(sys.argv) if j == '-test']
    for i in sys.argv:
        if i.isdigit():
            snapnumber = i
            break

    if len(test) > 0:
        model.test(snapnumber) # test model, the snapnumber is the number of the model snapshot
