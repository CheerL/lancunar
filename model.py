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
from itertools import product

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
        numpy_images = self.dataManagerValidation.numpy_images
        numpy_gts = self.dataManagerValidation.numpy_gts
        loss = 0.0
        accuracy = 0.0
        for key in numpy_images:
            _, temp_loss, temp_acc = self.produceSegmentationResult(model, numpy_images[key], numpy_gts[key])
            loss += temp_loss
            accuracy += temp_acc
        return loss / len(numpy_images), accuracy / len(numpy_images)

    def getTestResultImage(self, model, numpy_image, numpy_gt):
        result, loss, accuracy  = self.produceSegmentationResult(model, numpy_image, numpy_gt)
        self.logger.info("loss: {} acc: {}".format(loss, accuracy))
        return result

    def getTestResultImages(self, model):
        ''' return the segmentation results of the testing data'''
        numpy_images = self.dataManagerTest.numpy_images
        numpy_gts = self.dataManagerTest.numpy_gts
        ResultImages = dict()
        loss = 0.0
        accuracy = 0.0
        for key in numpy_images:
            temp_result, temp_loss, temp_acc = self.produceSegmentationResult(model, numpy_images[key], numpy_gts[key])
            loss += temp_loss
            accuracy += temp_acc
            ResultImages[key] = temp_result
        self.logger.info("loss: {} acc: {}".format(loss / len(numpy_images), accuracy / len(numpy_images)))
        return ResultImages

    def produceSegmentationResult(self, model, numpy_image, numpy_gt, call_loss=True):
        ''' produce the segmentation result, one time one image'''
        model.eval()
        # model.cuda()
        image_height, image_width, image_depth = image_shape = numpy_image.shape
        height, width, depth = vol_shape = self.params['DataManagerParams']['VolSize']
        stride_height, stride_width, stride_depth = self.params['DataManagerParams']['TestStride']

        result = np.zeros(image_shape, dtype=np.float32)
        result_weight = np.zeros(image_shape, dtype=np.float32)
        all_loss = list()

        # crop the image
        for ystart, xstart, zstart in product(
            list(range(0, image_height-height, stride_height)) + [image_height-height],
            list(range(0, image_width-width, stride_width)) + [image_width-width],
            list(range(0, image_depth-depth, stride_depth)) + [image_depth-depth]
        ):
            block_index = (
                slice(ystart, ystart + height),
                slice(xstart, xstart + width),
                slice(zstart, zstart + depth)
                )
            image_block = numpy_image[block_index].reshape(1, 1, *vol_shape)
            label_block = numpy_gt[block_index].reshape(1, 1, *vol_shape)

            data = torch.tensor(image_block).cuda().float()
            target = torch.tensor(label_block).cuda().int()
            target = target.view(-1)
            # volatile is used in the input variable for the inference,
            # which indicates the network doesn't need the gradients, and this flag will transfer to other variable
            # as the network computating
            output = model(data)
            pred = output.max(1)[1].view(tuple(vol_shape)).int()
            result[block_index] = result[block_index] + pred.cpu().numpy()
            result_weight[block_index] = result_weight[block_index] + 1

            if call_loss:
                block_loss = model.nll_loss(output, target).cpu().item()
                all_loss.append(block_loss)
        
        result = result / result_weight
        loss = np.mean(all_loss)
        accuracy = np.mean(result == numpy_gt)
        print(result.sum(), numpy_gt.sum())
        return result, loss, accuracy


    def save_checkpoint(self, state, path, prefix, filename='checkpoint.pth.tar'):
        ''' save the snapshot'''
        prefix_save = os.path.join(path, prefix)
        name = prefix_save + str(state['iteration']) + '_' + filename
        torch.save(state, name)

    def trainThread(self, model):
        '''train the network and plot the training curve'''
        nr_iter = self.params['ModelParams']['iteration']
        batchsize = self.params['ModelParams']['batchsize']
        snapshot = self.params['ModelParams']['snapshot']
        vol_shape = self.params['DataManagerParams']['VolSize']
        data_size = (batchsize, 1) + tuple(vol_shape)
        batch_image = np.zeros(data_size, dtype=self.dataManagerTrain.numpy_image_type)
        batch_label = np.zeros(data_size, dtype=self.dataManagerTrain.numpy_gt_type)

        train_interval = self.params['ModelParams']['trainInterval']
        val_interval = self.params['ModelParams']['valInterval']

        temp_loss = 0
        temp_accuracy = 0

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
                batch_image[i, 0], batch_label[i, 0] = self.dataManagerTrain.data_queue.get()
            print(batch_label.min(), batch_label.max())
            optimizer.zero_grad()
            data = torch.tensor(batch_image).cuda().float()
            target = torch.tensor(batch_label).cuda()
            target = target.view(-1)

            output = model(data)
            loss = model.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            pred = output.max(1)[1].view(-1).int()  # get the index of the max log-probability
            temp_loss += loss.cpu().item()
            temp_accuracy += pred.eq(target).float().mean().cpu().item()

            if not iteration % train_interval:
                train_accuracy = temp_accuracy / train_interval
                train_loss = temp_loss / train_interval
                self.logger.info(
                    "training: iter: {} loss: {} acc: {}".format(
                        snapshot + iteration,
                        train_loss,
                        train_accuracy
                    ))
                temp_accuracy = 0
                temp_loss = 0

            if not iteration % val_interval:
                val_loss, val_accuracy = self.getValidationLossAndAccuracy(model)
                if val_accuracy > self.max_accuracy:
                    self.max_accuracy = val_accuracy
                    self.max_accuracy_loss = val_loss
                    self.best_iteration_acc = snapshot + iteration
                    self.save_checkpoint({'iteration': self.best_iteration_acc,
                                          'state_dict': model.state_dict(),
                                          'best_acc': True},
                                          self.params['ModelParams']['dirSnapshots'],
                                          self.params['ModelParams']['tailSnapshots'])

                if val_loss < self.min_loss:
                    self.min_loss = val_loss
                    self.min_loss_accuracy = val_accuracy
                    self.best_iteration_loss = snapshot + iteration
                    self.save_checkpoint({'iteration': self.best_iteration_loss,
                                          'state_dict': model.state_dict(),
                                          'best_acc': False},
                                          self.params['ModelParams']['dirSnapshots'],
                                          self.params['ModelParams']['tailSnapshots'])
                    

                self.logger.info(
                    "validating: iteration: {} loss: {} accuracy: {}".format(
                        snapshot + iteration, val_loss, val_accuracy
                    ))
                self.logger.info(
                    "validating: best_acc: {} loss: {} accuracy: {}".format(
                        self.best_iteration_acc, self.max_accuracy_loss, self.max_accuracy
                ))
                self.logger.info(
                    "validating: best_loss: {} loss: {} accuracy: {}".format(
                        self.best_iteration_loss, self.min_loss, self.min_loss_accuracy
                ))

    def weights_init(self, m):
        ''' initialize the model'''
        classname = m.__class__.__name__
        if classname.find('Conv3d') != -1:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()

    def train(self, dataManagerTrain):
        ''' train model'''
        # we define here a data manager object
        self.logger.info('Start to train model')
        self.dataManagerTrain = dataManagerTrain
        self.dataManagerValidation = DMNII.DataManagerNii(
            self.params['ModelParams']['dirValidation'], 
            self.params['ModelParams']['dirResult'], 
            self.params['DataManagerParams']
            )
        self.dataManagerValidation.load_data()
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

        self.trainThread(model)

    def test(self, snapnumber):
        self.dataManagerTest = DMNII.DataManagerNii(
            self.params['ModelParams']['dirTest'],
            self.params['ModelParams']['dirResult'],
            self.params['DataManagerParams'],
        )
        self.dataManagerTest.load_data()
        model = Net(nll=True)
        prefix_save = os.path.join(
            self.params['ModelParams']['dirSnapshots'],
            self.params['ModelParams']['tailSnapshots']
        )
        name = prefix_save + str(snapnumber) + '_' + "checkpoint.pth.tar"
        checkpoint = torch.load(name)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        for name in tqdm(self.dataManagerTest.data_list):
            print(name)
            img = self.dataManagerTest.numpy_images[name]
            label = self.dataManagerTest.numpy_gts[name]
            result = self.getTestResultImage(model, img, label)
            self.dataManagerTest.write_result(result, name)
