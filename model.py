import sys
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
import os
import DataManager as DM
from os.path import splitext
from multiprocessing import Process, Queue
import math
import datetime
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import shutil
import vnet
from functools import reduce
import operator
import time
from tqdm import tqdm
import DataManagerNii as DMNII



class Model(object):

    ''' the network model for training, validation and testing '''
    params = None
    dataManagerTrain = None
    dataManagerValidation = None
    min_loss = 9999999999
    min_loss_accuracy = 0
    max_accuracy = 0
    max_accuracy_loss = 0
    best_iteration_acc = 0
    best_iteration_loss = 0

    def __init__(self, params):
        self.params = params

    def getValidationLossAndAccuracy(self, model):
        '''get the segmentation loss and accuracy of the validation data '''
        numpyImages = self.dataManagerValidation.numpyImages
        numpyGTs = self.dataManagerValidation.numpyGTs
        loss = 0.0
        accuracy = 0.0
        ResultImages = dict()
        for key in numpyImages:
            numpyResult, temploss = self.produceSegmentationResult(
                model, numpyImages[key], numpyGTs[key], calLoss=True
                )
            loss += temploss
            LabelResult = numpyResult

            '''cv2.imshow('0',LabelResult[:,:,32])
            cv2.waitKey(0)
            cv2.imshow('1',numpyGTs[keysIMG[i]][:,:,32])
            cv2.waitKey(0)'''
            right = float(np.sum(LabelResult == numpyGTs[key][:, :, :]))
            tot = float(LabelResult.shape[0] * LabelResult.shape[1] * LabelResult.shape[2])
            accuracy += right / tot
            ResultImages[key] = LabelResult
        return (loss / len(numpyImages), accuracy / len(numpyImages))

    def getTestResultImage(self, model, numpyImage, numpyGT, returnProbability=False):

        accuracy = 0.0
        (numpyResult, temploss) = self.produceSegmentationResult(
            model, numpyImage, numpyGT, calLoss=True, returnProbability=returnProbability)
        
        if returnProbability:
            LabelResult = numpyResult
        else:
            LabelResult = numpyResult
        '''cv2.imshow('0',LabelResult[:,:,32])
        cv2.waitKey(0)
        cv2.imshow('1',numpyGTs[keysIMG[i]][:,:,32])
        cv2.waitKey(0)'''
        right = float(np.sum(LabelResult == numpyGT[:, :, :]))
        tot = float(LabelResult.shape[0] * LabelResult.shape[1] * LabelResult.shape[2])
        accuracy += right / tot
        # plot_3d(numpyGT, threshold = 0)
        # plot_3d(LabelResult, threshold=0)
        print("loss: ", temploss , " acc: ", accuracy)
        return LabelResult

    def getTestResultImages(self, model, returnProbability=False):
        ''' return the segmentation results of the testing data'''
        numpyImages = self.dataManagerTesting.numpyImages
        numpyGTs = self.dataManagerTesting.numpyGTs
        loss = 0.0
        accuracy = 0.0
        ResultImages = dict()
        for key in numpyImages:
            (numpyResult, temploss) = self.produceSegmentationResult(
                model, numpyImages[key], numpyGTs[key], calLoss=True, returnProbability=returnProbability)
            loss += temploss
            if returnProbability:
                LabelResult = numpyResult
            else:
                LabelResult = numpyResult
            '''cv2.imshow('0',LabelResult[:,:,32])
            cv2.waitKey(0)
            cv2.imshow('1',numpyGTs[keysIMG[i]][:,:,32])
            cv2.waitKey(0)'''
            right = float(np.sum(LabelResult == numpyGTs[key][:, :, :]))
            tot = float(LabelResult.shape[0] * LabelResult.shape[1] * LabelResult.shape[2])
            accuracy += right / tot
            ResultImages[key] = LabelResult
        print("loss: ", loss / len(numpyImages), " acc: ", accuracy / len(numpyImages))
        return ResultImages

    def produceSegmentationResult(self, model, numpyImage, numpyGT=0, calLoss=False, returnProbability=False):
        ''' produce the segmentation result, one time one image'''
        model.eval()
        numpyImage = np.copy(numpyImage)
        padding_size = tuple((0, j - i % j) for i, j in zip(numpyImage.shape, self.params['DataManagerParams']['VolSize']))
        numpyImage = np.pad(numpyImage, padding_size, 'constant')
        numpyGT = np.pad(numpyGT, padding_size, 'constant')
        tempresult = np.zeros(numpyImage.shape, dtype=np.float32)
        tempWeight = np.zeros(numpyImage.shape, dtype=np.float32)
        height, width, depth = self.params['DataManagerParams']['VolSize']

        batchData = np.zeros([1, 1, height, width, depth])
        batchLabel = np.zeros([1, 1, height, width, depth])

        stride_height, stride_width, stride_depth = self.params['DataManagerParams']['TestStride']
        whole_height, whole_width, whole_depth = numpyImage.shape
        ynum, xnum, znum = [
            int(math.ceil((x - y) / float(z))) + 1
            for x, y, z in zip(
                numpyImage.shape,
                self.params['DataManagerParams']['VolSize'],
                self.params['DataManagerParams']['TestStride']
                )
            ]
        loss = 0
        acc = 0
        tot = 0
        # crop the image
        for y in range(ynum):
            for x in range(xnum):
                for z in range(znum):
                    if(y * stride_height + height < whole_height):
                        ystart = y * stride_height
                        yend = ystart + height
                    else:
                        ystart = whole_height - height
                        yend = whole_height
                    if(x * stride_width + width < whole_width):
                        xstart = x * stride_width
                        xend = xstart + width
                    else:
                        xstart = whole_width - width
                        xend = whole_width
                    if(z * stride_depth + depth < whole_depth):
                        zstart = z * stride_depth
                        zend = zstart + depth
                    else:
                        zstart = whole_depth - depth
                        if zstart < 0:
                            zend = whole_depth - zstart
                            zstart = 0
                        else:
                            zend = whole_depth

                    tot += 1
                    batchData[0, 0, :, :, :] = numpyImage[ystart:yend, xstart:xend, zstart:zend]
                    if calLoss:
                        batchLabel[0, 0, :, :, :] = numpyGT[ystart:yend, xstart:xend, zstart:zend]
                    else:
                        batchLabel[0, 0, :, :, :] = np.zeros(numpyImage[ystart:yend, xstart:xend, zstart:zend].shape)
                    data = torch.from_numpy(batchData).float()
                    # volatile is used in the input variable for the inference,
                    # which indicates the network doesn't need the gradients, and this flag will transfer to other variable
                    # as the network computating
                    data = Variable(data, volatile=True).cuda()
                    #data = Variable(data).cuda()
                    target = torch.from_numpy(batchLabel).long()
                    target = Variable(target).cuda()

                    original_shape = data[0, 0].size()
                    output = model(data)
                    target = target.view(target.numel())
                    temploss = F.nll_loss(output, target)
                    #temploss = bioloss.dice_loss(output, target)
                    # be carefull output is the log-probability, not the raw probability
                    # max(1) return a tumple,the second item is the index of the max
                    output = output.data.max(1)[1]
                    output = output.view(original_shape)
                    output = output.cpu()

                    temploss = temploss.cpu().item()
                    loss = loss + temploss
                    # print(temploss)
                    tempresult[ystart:yend, xstart:xend, zstart:zend] = tempresult[
                        ystart:yend, xstart:xend, zstart:zend] + output.numpy()
                    tempWeight[ystart:yend, xstart:xend, zstart:zend] = tempWeight[
                        ystart:yend, xstart:xend, zstart:zend] + 1
        tempresult = tempresult / tempWeight
        return (tempresult, loss)

    def create_temp_images(self, img, img_name):
        # create the 2D image based on the sitk image

        shape = img.shape
        n = math.sqrt(shape[0])
        n = int(n + 1)
        out_img = np.zeros([n * shape[1], n * shape[2]])
        img = (img * 255).astype(np.uint8)
        for i in range(n):
            for j in range(n):
                if i * n + j < shape[0]:
                    out_img[i * shape[1]:i * shape[1] + shape[1], j * shape[2]:j *
                            shape[2] + shape[2]] = img[i * n + j, :, :]

        #cv2.imwrite(os.path.join('tempImg', img_name + '.png'), out_img)

    

    def save_checkpoint(self, state, path, prefix, filename='checkpoint.pth.tar'):
        ''' save the snapshot'''
        prefix_save = os.path.join(path, prefix)
        name = prefix_save + str(state['iteration']) + '_' + filename
        torch.save(state, name)

    def trainThread(self, dataQueue, model):
        '''train the network and plot the training curve'''
        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']

        batchbasesize = (batchsize, 1) + tuple(self.params['DataManagerParams']['VolSize'])
        batchData = np.zeros(batchbasesize, dtype=float)
        batchLabel = np.zeros(batchbasesize, dtype=float)

        test_interval = self.params['ModelParams']['testInterval']
        train_interval = 50
        train_loss = np.zeros(nr_iter)
        train_accuracy = np.zeros(nr_iter // train_interval)
        testloss = np.zeros(nr_iter // test_interval)
        testaccuracy = np.zeros(nr_iter // test_interval)
        tempaccuracy = 0
        temptrain_loss = 0

        print("build vnet")

        model.train()
        model.cuda()
        optimizer = optim.Adam(
            model.parameters(),
            weight_decay=1e-8,
            lr=self.params['ModelParams']['baseLR']
            )

        for origin_it in range(nr_iter):
            it = origin_it + 1
            for i in range(batchsize):
                [defImg, defLab] = dataQueue.get()
                batchData[i, 0, :, :, :] = defImg
                batchLabel[i, 0, :, :, :] = defLab

            data = torch.from_numpy(batchData).float()
            data = Variable(data).cuda()
            target = torch.from_numpy(batchLabel).long()
            target = Variable(target).cuda()

            optimizer.zero_grad()
            output = model(data)
            target = target.view(target.numel())
            loss = F.nll_loss(output, target)
            #loss = bioloss.dice_loss(output, target)
            loss.backward()
            optimizer.step()

            temploss = loss.cpu().item()
            temptrain_loss = temptrain_loss + temploss
            # print(temptrain_loss)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            incorrect = pred.ne(target.data).cpu().sum()
            tempaccuracy = tempaccuracy + 1.0 - float(incorrect) / target.numel()

            if np.mod(it, train_interval) == 0:

                train_accuracy[it // train_interval] = tempaccuracy / (train_interval)
                train_loss[it // train_interval] = temptrain_loss / (train_interval)
                print(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    " training: iter: ", self.params['ModelParams']['snapshot'] + it,
                    " loss: ", train_loss[it / train_interval],
                    ' acc: ', train_accuracy[it / train_interval]
                )
               #plt.clf()
               # plt.subplot(2, 2, 1)
                #plt.plot(range(1, it // train_interval), train_loss[1:it // train_interval])
                #plt.subplot(2, 2, 2)
                #plt.plot(range(1, it // train_interval), train_accuracy[1:it // train_interval])
                #plt.subplot(2, 2, 3)
                #plt.plot(range(1, it // test_interval),
                 #        testloss[1:it // test_interval])
                #plt.subplot(2, 2, 4)
                #plt.plot(range(1, it // test_interval),
                 #        testaccuracy[1:it // test_interval])

                tempaccuracy = 0.0
                temptrain_loss = 0.0
                #plt.pause(0.00000001)

            if np.mod(it, test_interval) == 0:
                testloss[it // test_interval], testaccuracy[it // test_interval] = self.getValidationLossAndAccuracy(model)

                if testaccuracy[it // test_interval] >= self.max_accuracy:
                    self.max_accuracy = testaccuracy[it // test_interval]
                    self.min_accuracy_loss = testloss[it // test_interval]
                    self.best_iteration_acc = self.params['ModelParams']['snapshot'] + it
                    self.save_checkpoint({'iteration': self.params['ModelParams']['snapshot'] + it,
                                          'state_dict': model.state_dict(),
                                          'best_acc': True},
                                         self.params['ModelParams']['dirSnapshots'],
                                         self.params['ModelParams']['tailSnapshots'])

                if testloss[it // test_interval] <= self.min_loss:
                    self.min_loss = testloss[it // test_interval]
                    self.min_loss_accuracy = testaccuracy[it // test_interval]
                    self.best_iteration_loss = self.params['ModelParams']['snapshot'] + it
                    self.save_checkpoint({'iteration': self.params['ModelParams']['snapshot'] + it,
                                          'state_dict': model.state_dict(),
                                          'best_acc': False},
                                         self.params['ModelParams']['dirSnapshots'],
                                         self.params['ModelParams']['tailSnapshots'])

                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("\ntesting: best_acc: " + str(self.best_iteration_acc) +" loss: " + str(self.min_accuracy_loss) + " accuracy: " + str(self.max_accuracy))
                print("testing: best_loss: " + str(self.best_iteration_loss) + " loss: " + str(self.min_loss) + " accuracy: " + str(self.min_loss_accuracy))
                print("testing: iteration: " + str(self.params['ModelParams']['snapshot'] + it) + " loss: " + str(testloss[it / test_interval]) + " accuracy: " + str(testaccuracy[it / test_interval]) + '\n')
                #plt.clf()
                #plt.subplot(2, 2, 1)
                #plt.plot(range(1, it // 100), train_loss[1:it // 100])
                #plt.subplot(2, 2, 2)
                #plt.plot(range(1, it // 100), train_accuracy[1:it // 100])
                #plt.subplot(2, 2, 3)
                #plt.plot(range(1, it // test_interval),
                         #testloss[1:it // test_interval])
                #plt.subplot(2, 2, 4)
                #plt.plot(range(1, it // test_interval),
                     #    testaccuracy[1:it // test_interval])
                #plt.pause(0.00000001)

            #matplotlib.pyplot.show()

    def weights_init(self, m):
        ''' initialize the model'''
        classname = m.__class__.__name__
        if classname.find('Conv3d') != -1:
            nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()

    def train(self, dataManagerTrain, dataQueue):
        ''' train model'''
        # we define here a data manager object
        self.dataManagerTrain = dataManagerTrain
        self.dataManagerValidation = DM.DataManager(self.params['ModelParams']['dirValidation'], 
                                                    self.params['ModelParams']['dirResult'], 
                                                    self.params['DataManagerParams'])
        self.dataManagerValidation.loadTestData()

        howManyImages = len(self.dataManagerTrain.sitkImages)
        howManyGT = len(self.dataManagerTrain.sitkGT)

        assert howManyGT == howManyImages

        print("The dataset has shape: data - " + str(howManyImages) + ". labels - " + str(howManyGT))

        # create the network
        model = vnet.VNet(elu=False, nll=True)

        # train from scratch or continue from the snapshot
        if (self.params['ModelParams']['snapshot'] > 0):
            print("=> loading checkpoint ", str(self.params['ModelParams']['snapshot']))
            prefix_save = os.path.join(self.params['ModelParams']['dirSnapshots'],
                                       self.params['ModelParams']['tailSnapshots'])
            name = prefix_save + str(self.params['ModelParams']['snapshot']) + '_' + "checkpoint.pth.tar"
            checkpoint = torch.load(name)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint ", str(self.params['ModelParams']['snapshot']))
        else:
            model.apply(self.weights_init)

        #plt.ion()

        self.trainThread(dataQueue, model)



    def test(self, snapnumber):
        self.dataManagerTest = DMNII.DataManagerNiiLazyLoad(self.params['ModelParams']['dirTest'], 
                                                 self.params['ModelParams']['dirResult'], 
                                                 self.params['DataManagerParams'], 
                                                 self.params['TestParams']['ProbabilityMap'])
        self.dataManagerTest.loadTestData()
        model = vnet.VNet(elu=False, nll=False)
        prefix_save = os.path.join(self.params['ModelParams']['dirSnapshots'], self.params['ModelParams']['tailSnapshots'])
        name = prefix_save + str(snapnumber) + '_' + "checkpoint.pth.tar"
        checkpoint = torch.load(name)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        for f in tqdm(self.dataManagerTest.fileList):
            print(f)
            img, label = self.dataManagerTest.loadImgandLabel(f)
            result = self.getTestResultImage(model, img, label, self.params['TestParams']['ProbabilityMap'])
            self.dataManagerTest.writeResultsFromNumpyLabel(result,f)
            #self.dataManagerTest.writeResultsFromNumpyLabel(numpyImages[keysIMG[i]],keysIMG[i]+'_original',original_image=True)
