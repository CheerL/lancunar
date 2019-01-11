import sys
import os
import numpy as np
import model as model
import DataManagerNii as DMNII
from multiprocessing import Process, Queue


def prepareDataThread(proc, dataQueue, numpyImages, numpyGTs, params):
    ''' the thread worker to prepare the training data'''
    num_image = len(numpyImages)
    num_epoch = params['ModelParams']['epoch']
    batchsize = params['ModelParams']['batchsize']

    for num in range(proc, num_image * num_epoch, params['ModelParams']['nProc']):
        key = list(numpyImages.keys())[num % num_image]
        image = numpyImages[key].copy()
        gt = numpyGTs[key].copy()

        height, width, depth = params['DataManagerParams']['VolSize']
        stride_height, stride_width, stride_depth = params['DataManagerParams']['TrainStride']
        whole_height, whole_width, whole_depth = image.shape

        for ystart in list(range(0, whole_height-height, stride_height)) + [whole_height-height]:
            for xstart in list(range(0, whole_width-width, stride_width)) + [whole_width-width]:
                for zstart in list(range(0, whole_depth-depth, stride_depth)) + [whole_depth-depth]:
                    slice_index = (
                        slice(ystart, ystart + height),
                        slice(xstart, xstart + width),
                        slice(zstart, zstart + depth)
                    )
                    tempimage = image[slice_index]
                    tempGT = gt[slice_index]

                    # skip the image not containing the mircrobleed
                    # if tempGT.sum()<1:
                    #    continue
                    randomi = np.random.randint(4)
                    tempimage = np.rot90(tempimage, randomi)
                    tempGT = np.rot90(tempGT, randomi)
                    dataQueue.put((tempimage, tempGT))


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
    params['ModelParams']['dirTrain'] = '../CMB/training/' # the directory of training data
    # where we need to save the results (relative to the base path)
    params['ModelParams']['dirResult'] = "result/" # the directory of the results of testing data
    params['ModelParams']['dirValidation']='data/Lancunar/mini_val/' #the directory of the validation data
    # params['ModelParams']['dirTest']='data/Lancunar/Lacunar_testing/' #the directory of the testing data
    params['ModelParams']['dirTest']='../CMB/validation/' #the directory of the testing data
    # params['ModelParams']['dirResult']="/home/ftp/data/output/" #where we need to save the results (relative to the base path)
    # where to save the models while training
    params['ModelParams']['dirLog'] = "log/"
    params['ModelParams']['dirSnapshots'] = "snapshot/" # the directory of the model snapshots for training
    params['ModelParams']['tailSnapshots'] = 'WL/mini_vnet/' # the full path of the model snapshots is the join of dirsnapshots and presnapshots
    params['ModelParams']['batchsize'] = 64  # the batch size
    params['ModelParams']['epoch'] = 50000  # the number of total training iterations
    params['ModelParams']['baseLR'] = 1e-4  # the learning rate, initial one
    params['ModelParams']['weight_decay'] = 0.0005
    params['ModelParams']['nProc'] = 4  # the number of threads to do data augmentation
    params['ModelParams']['testInterval'] = 1000  # the number of training interations between testing
    params['ModelParams']['trainInterval'] = 50  # the number of training interations between testing

    # params of the DataManager
    params['DataManagerParams']['VolSize'] = np.asarray([64, 64, 8], dtype=int) # the size of the crop image
    params['DataManagerParams']['TestStride'] = np.asarray([64, 64, 8], dtype=int) # the stride of the adjacent crop image in testing phase and validation phase
    params['DataManagerParams']['TrainStride'] = np.asarray([32, 32, 4], dtype=int) # the stride of the adjacent crop image in testing phase and validation phase

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
        dataManagerTrain.loadData()  # loads in sitk format
        dataQueue = Queue(50)  # max 50 images in queue
        dataPreparation = [None] * params['ModelParams']['nProc']

        # thread creation
        for proc in range(0, params['ModelParams']['nProc']):
            dataPreparation[proc] = Process(
                target=prepareDataThread,
                args=(proc, dataQueue, dataManagerTrain.numpyImages, dataManagerTrain.numpyGTs, params)
            )
            dataPreparation[proc].daemon = True
            dataPreparation[proc].start()
        model.train(dataManagerTrain, dataQueue) #train model

    test = [i for i, j in enumerate(sys.argv) if j == '-test']
    for i in sys.argv:
        if i.isdigit():
            snapnumber = i
            break

    if len(test) > 0:
        model.test(snapnumber) # test model, the snapnumber is the number of the model snapshot
