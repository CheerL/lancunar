from __future__ import division, print_function

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import DataManagerNii as DMNII
from vnet import VNet as Net
from logger import Logger

class Model(object):
    ''' the network model for training, validation and testing '''
    dataManagerTrain = None
    dataManagerValidation = None
    dataManagerTest = None
    min_loss = 9999999999
    min_loss_accuracy = 0
    max_accuracy = 0
    max_accuracy_loss = 0
    best_iteration_acc = 0
    best_iteration_loss = 0

    def __init__(self, params):
        self.params = params
        self.logger = Logger(__name__, self.params['ModelParams']['dirLog'])

    def getValidationLossAndAccuracy(self, model):
        '''get the segmentation loss and accuracy of the validation data '''
        numpyImages = self.dataManagerValidation.numpyImages
        numpyGTs = self.dataManagerValidation.numpyGTs
        loss = 0.0
        accuracy = 0.0
        for key in numpyImages:
            _, temp_loss, temp_acc = self.produceSegmentationResult(model, numpyImages[key], numpyGTs[key])
            loss += temp_loss
            accuracy += temp_acc
        return loss / len(numpyImages), accuracy / len(numpyImages)

    def getTestResultImage(self, model, numpyImage, numpyGT):
        result, loss, accuracy  = self.produceSegmentationResult(model, numpyImage, numpyGT)
        self.logger.info("loss: {} acc: {}".format(loss, accuracy))
        return result

    def getTestResultImages(self, model):
        ''' return the segmentation results of the testing data'''
        numpyImages = self.dataManagerTest.numpyImages
        numpyGTs = self.dataManagerTest.numpyGTs
        ResultImages = dict()
        loss = 0.0
        accuracy = 0.0
        for key in numpyImages:
            temp_result, temp_loss, temp_acc = self.produceSegmentationResult(model, numpyImages[key], numpyGTs[key])
            loss += temp_loss
            accuracy += temp_acc
            ResultImages[key] = temp_result
        self.logger.info("loss: {} acc: {}".format(loss / len(numpyImages), accuracy / len(numpyImages)))
        return ResultImages

    def produceSegmentationResult(self, model, numpyImage, numpyGT, calLoss=True):
        ''' produce the segmentation result, one time one image'''
        # model.eval()
        # model.cuda()
        ori_shape = numpyImage.shape
        tempresult = np.zeros(numpyImage.shape, dtype=np.float32)
        tempWeight = np.zeros(numpyImage.shape, dtype=np.float32)
        height, width, depth = self.params['DataManagerParams']['VolSize']
        stride_height, stride_width, stride_depth = self.params['DataManagerParams']['TestStride']
        whole_height, whole_width, whole_depth = numpyImage.shape
        all_loss = list()
        # crop the image
        for ystart in list(range(0, whole_height-height, stride_height)) + [whole_height-height]:
            for xstart in list(range(0, whole_width-width, stride_width)) + [whole_width-width]:
                for zstart in list(range(0, whole_depth-depth, stride_depth)) + [whole_depth-depth]:
                    slice_index = (
                        slice(ystart, ystart + height),
                        slice(xstart, xstart + width),
                        slice(zstart, zstart + depth)
                        )
                    sliced_img = numpyImage[slice_index]
                    sliced_label = numpyGT[slice_index]
                    batchData = sliced_img.reshape(1, 1, *sliced_img.shape)
                    batchLabel = sliced_label.reshape(1, 1, *sliced_label.shape)
                    with torch.no_grad():
                        data = torch.tensor(batchData).cuda().float()
                        target = torch.tensor(batchLabel).cuda()
                    # volatile is used in the input variable for the inference,
                    # which indicates the network doesn't need the gradients, and this flag will transfer to other variable
                    # as the network computating
                    output = model(data)
                    pred = output.max(1)[1]

                    tempresult[slice_index] = tempresult[slice_index] + pred.cpu().numpy().reshape(*sliced_img.shape)
                    tempWeight[slice_index] = tempWeight[slice_index] + 1

                    if calLoss:
                        target = target.view(-1)
                        temploss = model.nll_loss(output, target).cpu().item()
                        all_loss.append(temploss)
        
        result = (tempresult / tempWeight)[:ori_shape[0], :ori_shape[1], :ori_shape[2]]
        loss = np.mean(all_loss)
        accuracy = np.mean(result == numpyGT)
        print(result.sum(), numpyGT.sum())
        return result, loss, accuracy


    def save_checkpoint(self, state, path, prefix, filename='checkpoint.pth.tar'):
        ''' save the snapshot'''
        prefix_save = os.path.join(path, prefix)
        name = prefix_save + str(state['iteration']) + '_' + filename
        torch.save(state, name)

    def trainThread(self, dataQueue, model):
        '''train the network and plot the training curve'''
        nr_iter = self.params['ModelParams']['epoch'] * self.dataManagerTrain.num
        batchsize = self.params['ModelParams']['batchsize']
        snapshot = self.params['ModelParams']['snapshot']
        batchbasesize = (batchsize, 1) + tuple(self.params['DataManagerParams']['VolSize'])
        batchData = np.zeros(batchbasesize, dtype=float)
        batchLabel = np.zeros(batchbasesize, dtype=float)

        test_interval = self.params['ModelParams']['testInterval']
        train_interval = self.params['ModelParams']['trainInterval']
        train_loss = np.zeros(nr_iter)
        train_accuracy = np.zeros(nr_iter // train_interval)
        testloss = np.zeros(nr_iter // test_interval)
        testaccuracy = np.zeros(nr_iter // test_interval)
        tempaccuracy = 0
        temptrain_loss = 0

        self.logger.info("Build V-Net")

        model.train()
        model.cuda()
        optimizer = optim.Adam(
            model.parameters(),
            weight_decay=self.params['ModelParams']['weight_decay'],
            lr=self.params['ModelParams']['baseLR']
        )

        for iteration in range(1, nr_iter+1):
            for i in range(batchsize):
                batchData[i, 0], batchLabel[i, 0] = dataQueue.get()

            data = torch.tensor(batchData).cuda().float()
            target = torch.tensor(batchLabel).cuda()
            target = target.view(-1)

            optimizer.zero_grad()
            output = model(data)
            pred = output.max(1)[1]  # get the index of the max log-probability
            loss = model.nll_loss(output, target)
            # loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            temptrain_loss += loss.cpu().item()
            tempaccuracy += pred.eq(target.long()).float().mean().cpu().item()

            if not iteration % train_interval:
                train_report_iter = iteration // train_interval - 1
                train_accuracy[train_report_iter] = tempaccuracy / train_interval
                train_loss[train_report_iter] = temptrain_loss / train_interval
                self.logger.info(
                    "training: iter: {} loss: {} acc: {}".format(
                        snapshot + iteration,
                        train_loss[train_report_iter],
                        train_accuracy[train_report_iter]
                    ))
                tempaccuracy = 0.0
                temptrain_loss = 0.0

            if not iteration % test_interval:
                test_report_iter = iteration // test_interval - 1
                testloss[test_report_iter], testaccuracy[test_report_iter] = self.getValidationLossAndAccuracy(model)
                if testaccuracy[test_report_iter] > self.max_accuracy:
                    self.max_accuracy = testaccuracy[test_report_iter]
                    self.max_accuracy_loss = testloss[test_report_iter]
                    self.best_iteration_acc = snapshot + iteration
                    self.save_checkpoint({'iteration': self.best_iteration_acc,
                                          'state_dict': model.state_dict(),
                                          'best_acc': True},
                                          self.params['ModelParams']['dirSnapshots'],
                                          self.params['ModelParams']['tailSnapshots'])
                    self.logger.info(
                        "testing: best_acc: {} loss: {} accuracy: {}".format(
                            self.best_iteration_acc, self.max_accuracy_loss, self.max_accuracy
                    ))

                if testloss[test_report_iter] < self.min_loss:
                    self.min_loss = testloss[test_report_iter]
                    self.min_loss_accuracy = testaccuracy[test_report_iter]
                    self.best_iteration_loss = snapshot + iteration
                    self.save_checkpoint({'iteration': self.best_iteration_loss,
                                          'state_dict': model.state_dict(),
                                          'best_acc': False},
                                          self.params['ModelParams']['dirSnapshots'],
                                          self.params['ModelParams']['tailSnapshots'])
                    self.logger.info(
                        "testing: best_loss: {} loss: {} accuracy: {}".format(
                            self.best_iteration_loss, self.min_loss, self.min_loss_accuracy
                    ))

                self.logger.info(
                    "testing: iteration: {} loss: {} accuracy: {}".format(
                        snapshot + iteration, testloss[test_report_iter], testaccuracy[test_report_iter]
                    ))

    def weights_init(self, m):
        ''' initialize the model'''
        classname = m.__class__.__name__
        if classname.find('Conv3d') != -1:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()

    def train(self, dataManagerTrain, dataQueue):
        ''' train model'''
        # we define here a data manager object
        self.logger.info('Start to train model')
        self.dataManagerTrain = dataManagerTrain
        self.dataManagerValidation = DMNII.DataManagerNii(
            self.params['ModelParams']['dirValidation'], 
            self.params['ModelParams']['dirResult'], 
            self.params['DataManagerParams']
            )
        self.dataManagerValidation.loadData()
        image_num = gt_num = self.dataManagerTrain.num
        self.logger.info("The dataset has shape: data {}. labels: {}".format(image_num, gt_num))
        # create the network
        model = Net(nll=True)

        # train from scratch or continue from the snapshot
        if self.params['ModelParams']['snapshot'] > 0:
            self.logger.info("loading checkpoint " + str(self.params['ModelParams']['snapshot']))
            prefix_save = os.path.join(
                self.params['ModelParams']['dirSnapshots'],
                self.params['ModelParams']['tailSnapshots']
            )
            name = prefix_save + str(self.params['ModelParams']['snapshot']) + '_' + "checkpoint.pth.tar"
            checkpoint = torch.load(name)
            model.load_state_dict(checkpoint['state_dict'])
            self.logger.info("loaded checkpoint " + str(self.params['ModelParams']['snapshot']))
        else:
            model.apply(self.weights_init)

        #plt.ion()

        self.trainThread(dataQueue, model)



    def test(self, snapnumber):
        self.dataManagerTest = DMNII.DataManagerNii(
            self.params['ModelParams']['dirTest'],
            self.params['ModelParams']['dirResult'],
            self.params['DataManagerParams'],
            self.params['TestParams']['ProbabilityMap']
        )
        self.dataManagerTest.loadData()
        model = Net()
        prefix_save = os.path.join(
            self.params['ModelParams']['dirSnapshots'],
            self.params['ModelParams']['tailSnapshots']
        )
        name = prefix_save + str(snapnumber) + '_' + "checkpoint.pth.tar"
        checkpoint = torch.load(name)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        for name in tqdm(self.dataManagerTest.fileList):
            print(name)
            img = self.dataManagerTest.numpyImages[name]
            label = self.dataManagerTest.numpyGTs[name]
            result = self.getTestResultImage(model, img, label)
            self.dataManagerTest.writeResultsFromNumpyLabel(result, name)
