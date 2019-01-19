from __future__ import division, print_function

import os
import time
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

import data_manager as data_manager
from logger import Logger
from vnet import VNet as Net


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
        self.logger = Logger(__name__, self.params['dirLog'])

    def all_predict(self, net, run_type='validation', silent=True, save=False):
        '''get the segmentation loss and accuracy of the validation data '''
        if run_type == 'validation':
            manager = self.dataManagerValidation
        elif run_type == 'testing':
            manager = self.dataManagerTest

        loss_list = list()
        accuracy_list = list()
        for data in manager.data_list:
            image = manager.numpy_images[data]
            gt = manager.numpy_gts[data]
            result, loss, accuracy = self.predict(net, manager, image, gt)
            loss_list.append(loss)
            accuracy_list.append(accuracy)

            if not silent:
                self.logger.info("{}: name: {} loss: {:.7f} acc: {:.5%} predict: {} gt: {}".format(
                    run_type, data, loss, accuracy, result.sum(), gt.sum()
                ))

            if save:
                manager.write_result(result, data)

        return np.mean(loss_list), np.mean(accuracy_list)

    def predict(self, net, manager, numpy_image, numpy_gt, call_loss=True):
        ''' produce the segmentation result, one time one image'''
        net.eval()
        vol_shape = manager.params['VolSize']
        result = np.zeros(numpy_gt.shape, dtype=manager.gt_feed_type)
        all_loss = list()

        # crop the image
        for num, (image_block, gt_block) in enumerate(zip(numpy_image, numpy_gt)):
            image_block = image_block.reshape(1, 1, *vol_shape)
            gt_block = gt_block.reshape(1, 1, *vol_shape)
        # batch_size = manager.params['batchsize']
        # for num in range(0, numpy_gt.shape[0], batch_size):
        #     image_block = numpy_image[num:num+batch_size].reshape(-1, 1, *vol_shape)
        #     gt_block = numpy_gt[num:num+batch_size].reshape(-1, 1, *vol_shape)

            data = torch.tensor(image_block).cuda().float()
            target = torch.tensor(gt_block).cuda().view(-1)
            # volatile is used in the input variable for the inference,
            # which indicates the network doesn't need the gradients, and this flag will transfer to other variable
            # as the network computating
            output = net(data)
            pred = output.max(1)[1].view(*vol_shape)
            result[num] = pred.cpu().numpy()
            # pred = output.max(1)[1].view(-1, *vol_shape)
            # result[num:num+batch_size] = pred.cpu().numpy()

            if call_loss:
                block_loss = net.loss(output, target).cpu().item()
                all_loss.append(block_loss)

        loss = np.mean(all_loss)
        accuracy = np.mean(result == numpy_gt)
        return result, loss, accuracy


    def save_checkpoint(self, state, path, prefix, filename='checkpoint.pth.tar'):
        ''' save the snapshot'''
        prefix_save = os.path.join(path, prefix)
        name = prefix_save + str(state['iteration']) + '_' + filename
        torch.save(state, name)

    def weights_init(self, m):
        ''' initialize the model'''
        classname = m.__class__.__name__
        if classname.find('Conv3d') != -1:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()

    def train(self):
        ''' train model'''
        nr_iter = self.params['iteration']
        snapshot = self.params['snapshot']
        train_interval = self.params['trainInterval']
        val_interval = self.params['valInterval']

        self.dataManagerTrain = data_manager.DataManager(
            self.params['dirTrain'], 
            self.params['dirResult'],
            self.params['dataManager']
        )
        self.dataManagerValidation = data_manager.DataManager(
            self.params['dirValidation'], 
            self.params['dirResult'],
            self.params['dataManager']
        )
        self.dataManagerTrain.load_data()
        self.dataManagerValidation.load_data()

        self.dataManagerTrain.run_feed_thread()
        # create the network
        net = Net(loss_type=self.params['loss'])
        if self.params['snapshot'] > 0:
            self.logger.info("loading checkpoint " + str(self.params['snapshot']))
            prefix_save = os.path.join(
                self.params['dirSnapshots'],
                self.params['tailSnapshots']
            )
            name = prefix_save + str(self.params['snapshot']) + '_' + "checkpoint.pth.tar"
            checkpoint = torch.load(name)
            net.load_state_dict(checkpoint['state_dict'])
            self.logger.info("loaded checkpoint " + str(self.params['snapshot']))
        else:
            net.apply(self.weights_init)

        temp_loss = 0
        temp_accuracy = 0

        net.train()
        net.cuda()

        optimizer = optim.Adam(
            net.parameters(),
            weight_decay=self.params['weight_decay'],
            lr=self.params['baseLR']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=25, verbose=True)
        self.logger.info("Run {}".format(net.net_name))

        for iteration in range(1, nr_iter+1):
            batch_image, batch_gt = self.dataManagerTrain.data_queue.get()
            data = torch.tensor(batch_image).cuda().float()
            target = torch.tensor(batch_gt).cuda().view(-1)

            output = net(data)

            loss = net.loss(output, target)
            flatten = output.max(1)[1].view(-1)  # get the index of the max log-probability
            temp_loss += loss.cpu().item()
            temp_accuracy += flatten.eq(target).float().mean().cpu().item()

            real_iteration = snapshot + iteration
            if not iteration % train_interval:
                train_accuracy = temp_accuracy / train_interval
                train_loss = temp_loss / train_interval
                self.logger.info(
                    "training: iter: {} loss: {:.7f} acc: {:.5%}".format(
                        real_iteration,
                        train_loss,
                        train_accuracy
                    ))
                temp_accuracy = 0
                temp_loss = 0

            if not iteration % val_interval:
                val_loss, val_accuracy = self.all_predict(net, silent=False)
                if val_accuracy > self.max_accuracy:
                    self.max_accuracy = val_accuracy
                    self.max_accuracy_loss = val_loss
                    self.best_iteration_acc = real_iteration

                if val_loss < self.min_loss:
                    self.min_loss = val_loss
                    self.min_loss_accuracy = val_accuracy
                    self.best_iteration_loss = real_iteration

                self.save_checkpoint({'iteration': real_iteration,
                                        'state_dict': net.state_dict(),
                                        'best_acc': self.best_iteration_loss == real_iteration},
                                        self.params['dirSnapshots'],
                                        self.params['tailSnapshots'])

                self.logger.info(
                    "validating: iteration: {} loss: {:.7f} accuracy: {:.5%}".format(
                        real_iteration, val_loss, val_accuracy
                ))
                self.logger.info(
                    "validating: best_acc: {} loss: {:.7f} accuracy: {:.5%}".format(
                        self.best_iteration_acc, self.max_accuracy_loss, self.max_accuracy
                ))
                self.logger.info(
                    "validating: best_loss: {} loss: {:.7f} accuracy: {:.5%}".format(
                        self.best_iteration_loss, self.min_loss, self.min_loss_accuracy
                ))

            optimizer.zero_grad()
            loss.backward()
            scheduler.step(loss)
            # optimizer.step()

    def test(self):
        self.dataManagerTest = data_manager.DataManager(
            self.params['dirTest'],
            self.params['dirResult'],
            self.params['dataManager'],
        )
        self.dataManagerTest.load_data()

        net = Net(loss_type=self.params['loss'])
        prefix_save = os.path.join(
            self.params['dirSnapshots'],
            self.params['tailSnapshots']
        )
        name = prefix_save + str(self.params['snapshot']) + '_' + "checkpoint.pth.tar"
        checkpoint = torch.load(name)
        net.load_state_dict(checkpoint['state_dict'])
        net.cuda()
        loss, accuracy = self.all_predict(net, run_type='testing', silent=False, save=False)
        self.logger.info('testing: loss: {:.7f} accuracy: {:.5%}'.format(loss, accuracy))
