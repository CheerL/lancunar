import sys
import os
import numpy as np
import model as model
import DataManager as DM
from multiprocessing import Process, Queue
def prepareDataThread( dataQueue, numpyImages, numpyGT, params):
    ''' the thread worker to prepare the training data'''
    nr_iter = params['ModelParams']['numIterations']
    batchsize = params['ModelParams']['batchsize']

    keysIMG = list(numpyImages.keys())

    nr_iter_dataAug = nr_iter * batchsize
    np.random.seed()
    whichDataList = np.random.randint(len(keysIMG), size=int(nr_iter_dataAug / params['ModelParams']['nProc']))
    whichDataForMatchingList = np.random.randint(len(keysIMG), size=int(
        nr_iter_dataAug / params['ModelParams']['nProc']))

    last_blood = False
    for whichData, whichDataForMatching in zip(whichDataList, whichDataForMatchingList):
        currGtKey = keysIMG[whichData]
        currImgKey = keysIMG[whichData]
        tempimage = np.copy(numpyImages[currImgKey])
        tempGT = np.copy(numpyGT[currGtKey])

        # data agugumentation through hist matching across different examples...
        # this data augmentation uses too much time
        ImgKeyMatching = keysIMG[whichDataForMatching]
        #tempimage = utilities.hist_match(tempimage, numpyImages[ImgKeyMatching])

        '''if(np.random.rand(1)[0]>0.5): #do not apply deformations always, just sometimes
            tempimage, tempGT = utilities.params['TestParams']produceRandomlyDeformedImage(tempimage, tempGT,
                                self.params['ModelParams']['numcontrolpoints'],
                                            self.params['ModelParams']['sigma'])'''
        # image_height, image_width, image_depth = tempimage.shape
        # starty = np.random.randint(image_height - params['DataManagerParams']['VolSize'][0])
        # startx = np.random.randint(image_width - params['DataManagerParams']['VolSize'][1])
        # startz = np.random.randint(image_depth - params['DataManagerParams']['VolSize'][2])
        starty, startx, startz = [
            np.random.randint(i - j) if i > j else 0
            for i, j in zip(tempimage.shape, params['DataManagerParams']['VolSize'])
            ]
        tempimage = tempimage[
            starty: starty + params['DataManagerParams']['VolSize'][0],
            startx: startx + params['DataManagerParams']['VolSize'][1],
            startz: startz + params['DataManagerParams']['VolSize'][2]
            ]
        tempGT = tempGT[
            starty: starty + params['DataManagerParams']['VolSize'][0],
            startx: startx + params['DataManagerParams']['VolSize'][1],
            startz: startz + params['DataManagerParams']['VolSize'][2]
            ]
        padding_size = tuple(
            (0, j - i) if j > i else (0, 0)
            for i, j in zip(tempimage.shape, params['DataManagerParams']['VolSize']))
        tempimage = np.pad(tempimage, padding_size, 'constant').astype(dtype=np.float32)
        tempGT = np.pad(tempGT, padding_size, 'constant').astype(dtype=np.float32)

        # skip the image not containing the mircrobleed
        # if tempGT.sum()<1:
        #    continue
        if tempGT.sum() < 1 and not last_blood:
            continue
        if tempGT.sum() < 1:
            last_blood = False
        else:
            last_blood = True
        randomi = np.random.randint(4)
        tempimage = np.rot90(tempimage, randomi)
        tempGT = np.rot90(tempGT, randomi)
        tempimage = tempimage * (np.random.rand(1) + 0.5)

        dataQueue.put(tuple((tempimage, tempGT)))



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
    # where we need to save the results (relative to the base path)
    params['ModelParams']['dirResult'] = "result/" # the directory of the results of testing data
    params['ModelParams']['dirValidation']='data/Lancunar/Lacunar_validation/' #the directory of the validation data
    params['ModelParams']['dirTest']='data/Lancunar/Lacunar_testing/' #the directory of the testing data
    # params['ModelParams']['dirResult']="/home/ftp/data/output/" #where we need to save the results (relative to the base path)
    # where to save the models while training
    params['ModelParams']['dirSnapshots'] = "snapshot/" # the directory of the model snapshots for training
    params['ModelParams']['tailSnapshots'] = 'WL/vnet/' # the full path of the model snapshots is the join of dirsnapshots and presnapshots
    params['ModelParams']['batchsize'] = 2  # the batch size
    params['ModelParams']['numIterations'] = 20000  # the number of total training iterations
    params['ModelParams']['baseLR'] = 0.0003  # the learning rate, initial one
    params['ModelParams']['nProc'] = 5  # the number of threads to do data augmentation
    params['ModelParams']['testInterval'] = 100  # the number of training interations between testing


    # params of the DataManager
    params['DataManagerParams']['VolSize'] = np.asarray([64, 64, 24], dtype=int) # the size of the crop image
    params['DataManagerParams']['TestStride'] = np.asarray([64, 64, 24], dtype=int) # the stride of the adjacent crop image in testing phase and validation phase

    # Ture: produce the probaility map in the testing phase, False: produce the  label image
    params['TestParams']['ProbabilityMap'] = False
    model = model.Model(params)
    train = [i for i, j in enumerate(sys.argv) if j == '-train']
    if len(train) > 0:
        dataManagerTrain = DM.DataManager(params['ModelParams']['dirTrain'],
                                          params['ModelParams']['dirResult'],
                                          params['DataManagerParams'])

        dataManagerTrain.loadTrainingData()  # loads in sitk format
        howManyImages = len(dataManagerTrain.sitkImages)
        howManyGT = len(dataManagerTrain.sitkGT)

        assert howManyGT == howManyImages
        numpyImages = dataManagerTrain.getNumpyImages()
        numpyGT = dataManagerTrain.getNumpyGT()
        dataQueue = Queue(50)  # max 50 images in queue
        dataPreparation = [None] * params['ModelParams']['nProc']

        # thread creation
        for proc in range(0, params['ModelParams']['nProc']):
            dataPreparation[proc] = Process(target=prepareDataThread, args=(dataQueue, numpyImages, numpyGT, params))
            dataPreparation[proc].daemon = True
            dataPreparation[proc].start()

        model.train(dataManagerTrain, dataQueue) #train model

    test = [i for i, j in enumerate(sys.argv) if j == '-test']
    for i in sys.argv:
        if(i.isdigit()):
            snapnumber = i
            break
    if len(test) > 0:
        model.test(snapnumber) # test model, the snapnumber is the number of the model snapshot
