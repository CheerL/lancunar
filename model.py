from __future__ import print_function
from __future__ import division

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
    dataManagerTesting = None
    min_loss = 9999999999
    min_loss_accuracy = 0
    max_accuracy = 0
    max_accuracy_loss = 0
    best_iteration_acc = 0
    best_iteration_loss = 0

    def __init__(self, params):
        self.params = params

    def predict(self, model, numpyImage, numpyGT):
        result, loss = self.produceSegmentationResult(model, numpyImage, numpyGT, calLoss=True)
        accuracy = np.sum(result == numpyGT) / result.size
        return result, accuracy, loss

    def getValidationLossAndAccuracy(self, model):
        '''get the segmentation loss and accuracy of the validation data '''
        numpyImages = self.dataManagerValidation.numpyImages
        numpyGTs = self.dataManagerValidation.numpyGTs
        loss = 0.0
        accuracy = 0.0
        for key in numpyImages:
            _, temp_loss, temp_acc = self.predict(model, numpyImages[key], numpyGTs[key])
            loss += temp_loss
            accuracy += temp_acc
        return loss / len(numpyImages), accuracy / len(numpyImages)

    def getTestResultImage(self, model, numpyImage, numpyGT):
        result, accuracy, loss = self.predict(model, numpyImage, numpyGT)
        print("loss: {} acc: {}".format(loss, accuracy))
        return result

    def getTestResultImages(self, model):
        ''' return the segmentation results of the testing data'''
        numpyImages = self.dataManagerTesting.numpyImages
        numpyGTs = self.dataManagerTesting.numpyGTs
        ResultImages = dict()
        loss = 0.0
        accuracy = 0.0
        for key in numpyImages:
            temp_result, temp_loss, temp_acc = self.predict(model, numpyImages[key], numpyGTs[key])
            loss += temp_loss
            accuracy += temp_acc
            ResultImages[key] = temp_result
        print("loss: {} acc: {}".format(loss / len(numpyImages), accuracy / len(numpyImages)))
        return ResultImages

    def produceSegmentationResult(self, model, numpyImage, numpyGT=0, calLoss=False):
        ''' produce the segmentation result, one time one image'''
        model.eval()
        padding_size = tuple(
            (0, j - i % j if i % j else 0)
            for i, j in zip(numpyImage.shape, self.params['DataManagerParams']['VolSize'])
            )
        numpyImage = np.pad(numpyImage, padding_size, 'constant')
        tempresult = np.zeros(numpyImage.shape, dtype=np.float32)
        tempWeight = np.zeros(numpyImage.shape, dtype=np.float32)
        height, width, depth = self.params['DataManagerParams']['VolSize']
        stride_height, stride_width, stride_depth = self.params['DataManagerParams']['TestStride']
        whole_height, whole_width, whole_depth = numpyImage.shape
        loss = 0
        # crop the image
        for ystart in range(0, whole_height, stride_height):
            for xstart in range(0, whole_width, stride_width):
                for zstart in range(0, whole_depth, stride_depth):
                    slice_index = (
                        slice(ystart, ystart + height),
                        slice(xstart, xstart + width),
                        slice(zstart, zstart + depth)
                        )
                    sliced_img = numpyImage[slice_index]
                    batchData = sliced_img.reshape(1, 1, *sliced_img.shape)
                    data = torch.from_numpy(batchData).float()
                    # volatile is used in the input variable for the inference,
                    # which indicates the network doesn't need the gradients, and this flag will transfer to other variable
                    # as the network computating
                    data = Variable(data).cuda()
                    output = model(data)
                    pred = output.data.max(1)[1].cpu()

                    tempresult[slice_index] = tempresult[slice_index] + pred.numpy().reshape(*sliced_img.shape)
                    tempWeight[slice_index] = tempWeight[slice_index] + 1

                    if calLoss:
                        numpyGT = np.pad(numpyGT, padding_size, 'constant')
                        sliced_label = numpyGT[slice_index]
                        batchLabel = sliced_label.reshape(1, 1, *sliced_label.shape)
                        target = torch.from_numpy(batchLabel).long()
                        target = Variable(target).cuda()
                        target = target.view(target.numel())
                        temploss = F.nll_loss(output, target).cpu().item()
                        loss += temploss
        tempresult = tempresult / tempWeight
        return tempresult, loss

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
            for i in range(batchsize):
                [defImg, defLab] = dataQueue.get()
                batchData[i, 0] = defImg
                batchLabel[i, 0] = defLab

            data = torch.from_numpy(batchData).float()
            data = Variable(data).cuda()
            target = torch.from_numpy(batchLabel).long()
            target = Variable(target).cuda()
            target = target.view(target.numel())

            optimizer.zero_grad()
            output = model(data)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            loss = F.nll_loss(output, target)
            #loss = bioloss.dice_loss(output, target)
            loss.backward()
            optimizer.step()

            temptrain_loss += loss.cpu().item()
            tempaccuracy += pred.eq(target.data).cpu().sum() / target.numel()

            it = origin_it + 1
            if not it % train_interval:
                train_report_it = it // train_interval
                train_accuracy[train_report_it] = tempaccuracy / train_interval
                train_loss[train_report_it] = temptrain_loss / train_interval
                print(
                    "{} training: iter: {} loss: {} acc: {}".format( 
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        self.params['ModelParams']['snapshot'] + it,
                        train_loss[train_report_it],
                        train_accuracy[train_report_it]
                    )
                )
                tempaccuracy = 0.0
                temptrain_loss = 0.0

            if not it % test_interval:
                test_report_it = it // test_interval
                save_it = self.params['ModelParams']['snapshot'] + it
                testloss[test_report_it], testaccuracy[test_report_it] = self.getValidationLossAndAccuracy(model)

                if testaccuracy[test_report_it] > self.max_accuracy:
                    self.max_accuracy = testaccuracy[test_report_it]
                    self.max_accuracy_loss = testloss[test_report_it]
                    self.best_iteration_acc = save_it
                    self.save_checkpoint({'iteration': save_it,
                                          'state_dict': model.state_dict(),
                                          'best_acc': True},
                                         self.params['ModelParams']['dirSnapshots'],
                                         self.params['ModelParams']['tailSnapshots'])

                if testloss[test_report_it] < self.min_loss:
                    self.min_loss = testloss[test_report_it]
                    self.min_loss_accuracy = testaccuracy[test_report_it]
                    self.best_iteration_loss = save_it
                    self.save_checkpoint({'iteration': save_it,
                                          'state_dict': model.state_dict(),
                                          'best_acc': False},
                                         self.params['ModelParams']['dirSnapshots'],
                                         self.params['ModelParams']['tailSnapshots'])

                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("testing: best_acc: {} loss: {} accuracy: {}".format(self.best_iteration_acc, self.min_accuracy_loss, self.max_accuracy))
                print("testing: best_loss: {} loss: {} accuracy: {}".format(self.best_iteration_loss, self.min_loss, self.min_loss_accuracy))
                print("testing: iteration: {} loss: {} accuracy: {}".format(save_it, testloss[test_report_it], testaccuracy[test_report_it]))

    def weights_init(self, m):
        ''' initialize the model'''
        classname = m.__class__.__name__
        if classname.find('Conv3d') != -1:
            nn.init.kaiming_normal_(m.weight)
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
        print("The dataset has shape: data {}. labels: {}".format(howManyImages, howManyGT))
        # create the network
        model = vnet.VNet(elu=False, nll=True)

        # train from scratch or continue from the snapshot
        if (self.params['ModelParams']['snapshot'] > 0):
            print("=> loading checkpoint " + str(self.params['ModelParams']['snapshot']))
            prefix_save = os.path.join(self.params['ModelParams']['dirSnapshots'],
                                       self.params['ModelParams']['tailSnapshots'])
            name = prefix_save + str(self.params['ModelParams']['snapshot']) + '_' + "checkpoint.pth.tar"
            checkpoint = torch.load(name)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint " + str(self.params['ModelParams']['snapshot']))
        else:
            model.apply(self.weights_init)

        #plt.ion()

        self.trainThread(dataQueue, model)



    def test(self, snapnumber):
        self.dataManagerTest = DMNII.DataManagerNiiLazyLoad(
            self.params['ModelParams']['dirTest'],
            self.params['ModelParams']['dirResult'],
            self.params['DataManagerParams'],
            self.params['TestParams']['ProbabilityMap']
        )
        self.dataManagerTest.loadTestData()
        model = vnet.VNet(elu=False, nll=False)
        prefix_save = os.path.join(
            self.params['ModelParams']['dirSnapshots'],
            self.params['ModelParams']['tailSnapshots']
        )
        name = prefix_save + str(snapnumber) + '_' + "checkpoint.pth.tar"
        checkpoint = torch.load(name)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        for file in tqdm(self.dataManagerTest.fileList):
            print(file)
            img, label = self.dataManagerTest.loadImgandLabel(file)
            result = self.getTestResultImage(model, img, label)
            self.dataManagerTest.writeResultsFromNumpyLabel(result, file)
