from __future__ import division, print_function

import os
import time
from itertools import product
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

# from data_manager import DataManager
from data_manager import DataManager2D as DataManager
from logger import Logger
# from vnet import VNet as Net
from unet import UNet as Net
import matplotlib.pyplot as plt


class RewarmCosineAnnealingLR(optim.lr_scheduler.CosineAnnealingLR):
    def get_period(self):
        return int(math.log2(self.last_epoch / self.T_max + 1))

    def get_lr(self):
        period = self.get_period()
        T_period = self.T_max ** period
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch - T_period + 1) / T_period)) / 2
                for base_lr in self.base_lrs]

class Model:
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
        iou_list = list()
        for data in manager.data_list:
            image = manager.numpy_images[data]
            gt = manager.numpy_gts[data]
            result, loss, accuracy, iou = self.predict(net, manager, image, gt)
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            iou_list.append(iou)

            if not silent:
                self.logger.info("{}: name: {} loss: {:.7f} acc: {:.5%}, iou: {:.5%} predict: {} gt: {}".format(
                    run_type, data, loss, accuracy, iou, result.sum(), gt.sum()
                ))

            if save:
                manager.write_result(result, data)

        return np.mean(loss_list), np.mean(accuracy_list), np.mean(iou_list)

    def predict(self, net, manager, numpy_image, numpy_gt, call_loss=True):
        ''' produce the segmentation result, one time one image'''
        net.eval()
        vol_shape = manager.params['VolSize']
        result = np.zeros(numpy_gt.shape, dtype=manager.gt_feed_type)
        all_loss = list()
        all_iou = list()
        batch_size = manager.params['batchsize']
        size = len(numpy_image)
        # crop the image
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            image_block = numpy_image[start:end].reshape(-1, 1, *vol_shape)
            gt_block = numpy_gt[start:end].reshape(-1, 1, *vol_shape)
        # for num, (image_block, gt_block) in enumerate(zip(numpy_image, numpy_gt)):
            # image_block = image_block.reshape(1, 1, *vol_shape)
            # gt_block = gt_block.reshape(1, 1, *vol_shape)

            data = torch.Tensor(image_block).cuda().float()
            target = torch.Tensor(gt_block).cuda().long().view(-1)
            # volatile is used in the input variable for the inference,
            # which indicates the network doesn't need the gradients,
            # and this flag will transfer to other variable as the network computating
            output = net(data)
            pred = output.max(1)[1].view(-1, *vol_shape)
            result[start:end] = pred.cpu().numpy()
            # pred = output.max(1)[1].view(-1, *vol_shape)
            # result[num:num+batch_size] = pred.cpu().numpy()
            block_iou = net.iou(output, target).cpu().item()

            if gt_block.any():
                all_iou.append(block_iou)

            if call_loss:
                block_loss = net.loss(output, target).cpu().item()
                all_loss.append(block_loss)

        iou = np.mean(all_iou) if all_iou else 0
        loss = np.mean(all_loss)
        accuracy = np.mean(result == numpy_gt)
        return result, loss, accuracy, iou


    def save_checkpoint(self, state, path, prefix, filename='checkpoint.pth.tar'):
        ''' save the snapshot'''
        prefix_save = os.path.join(path, prefix)
        name = prefix_save + str(state['iteration']) + '_' + filename
        torch.save(state, name)

    def _weights_init(self, m):
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

        self.dataManagerTrain = DataManager(
            self.params['dirTrain'],
            self.params['dirResult'],
            self.params['dataManager']
        )
        self.dataManagerValidation = DataManager(
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
            net.apply(self._weights_init)

        temp_loss = 0
        temp_accuracy = 0
        temp_iou = 0
        # optimizer = optim.Adam(
        #     net.parameters(),
        #     weight_decay=self.params['weight_decay'],
        #     lr=self.params['baseLR']
        # )
        optimizer = optim.SGD(
            net.parameters(),
            lr=self.params['baseLR'],
            momentum=0.9,
            weight_decay=self.params['weight_decay'],
        )

        net.train()
        net.cuda()

        t_max = max(
            200, round(self.dataManagerTrain.data_num / self.dataManagerTrain.batch_size * 8, 2)
        )
        scheduler = RewarmCosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=self.params['minLR']
        )
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.params['expgamma'])
        scheduler.last_epoch = snapshot - 1

        self.logger.info('Create scheduler max lr: {} min lr: {} Tmax: {}'.format(
            self.params['baseLR'], self.params['minLR'], t_max
        ))
        self.logger.info('Run {}'.format(net.net_name))

        for iteration in range(1, nr_iter+1):
            batch_image, batch_gt = self.dataManagerTrain.data_queue.get()
            data = torch.Tensor(batch_image).cuda().float()
            target = torch.Tensor(batch_gt).cuda().long().view(-1)
            output = net(data)
            loss = net.loss(output, target)
            flatten = output.max(1)[1].view(-1)
            temp_loss += loss.cpu().item()
            temp_accuracy += flatten.eq(target).float().mean().cpu().item()
            temp_iou += net.iou(output, target)

            real_iteration = snapshot + iteration
            if not iteration % train_interval:
                train_accuracy = temp_accuracy / train_interval
                train_loss = temp_loss / train_interval
                train_iou = temp_iou / train_interval
                self.logger.info(
                    "training: iter: {} loss: {:.7f} acc: {:.5%}, iou: {:.5%}".format(
                        real_iteration,
                        train_loss,
                        train_accuracy,
                        train_iou
                    ))
                temp_accuracy = 0
                temp_loss = 0
                temp_iou = 0

            # if not iteration % val_interval or iteration is 1:
            if not iteration % val_interval:
                val_loss, val_accuracy, val_iou = self.all_predict(net, silent=False)
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
                    "validating: iteration: {} loss: {:.7f} accuracy: {:.5%}, iou: {:.5%}".format(
                        real_iteration, val_loss, val_accuracy, val_iou
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
            scheduler.step()
            optimizer.step()

    def test(self):
        self.dataManagerTest = DataManager(
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
        loss, accuracy, iou = self.all_predict(net, run_type='testing', silent=False, save=True)
        self.logger.info('testing: loss: {:.7f}, accuracy: {:.5%}, iou: {:.5%}'.format(loss, accuracy, iou))

    def _try_lr(self, net, optimizer, start_lr=1e-7, end_lr=10.0, num=100, beta=0.9):
        def update_lr(optimizer, lr):
            for group in optimizer.param_groups:
                group['lr'] = lr

        factor = (end_lr / start_lr) ** (1 / num)
        lr = start_lr
        avg_loss = 0.
        best_loss = 0.
        losses = []
        log_lrs = []

        for iteration in range(1, num+1):
            update_lr(optimizer, lr)
            batch_image, batch_gt = self.dataManagerTrain.data_queue.get()
            data = torch.Tensor(batch_image).cuda().float()
            target = torch.Tensor(batch_gt).cuda().long().view(-1)

            output = net(data)
            loss = net.loss(output, target)

            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.cpu().item()
            smoothed_loss = avg_loss / (1 - beta ** iteration)
            #Stop if the loss is exploding
            if iteration > 1 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or iteration == 1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lr = np.log10(lr)
            log_lrs.append(log_lr)
            #Do the SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #Update the lr for the next step
            self.logger.info(
                'lr: {}, log lr: {:.4f}, loss: {:.7f}, best loss: {:.7f}'.format(
                    lr, log_lr, smoothed_loss, best_loss
                ))
            lr *= factor
            if not iteration % 10 or iteration == num:
                plt.plot(log_lrs, losses)
                plt.savefig('find_lr.{}.{}.jpg'.format(start_lr, end_lr))

    def find_lr(self, start_lr=1e-7, end_lr=10.0, num=100, beta=0.9):
        self.dataManagerTrain = DataManager(
            self.params['dirTrain'],
            self.params['dirResult'],
            self.params['dataManager']
        )
        self.dataManagerTrain.load_data()
        self.dataManagerTrain.run_feed_thread()
        # create the network
        net = Net(loss_type=self.params['loss'])
        net.apply(self._weights_init)

        net.train()
        net.cuda()

        optimizer = optim.SGD(
            net.parameters(),
            lr=self.params['baseLR'],
            momentum=0.9,
            weight_decay=self.params['weight_decay'],
        )
        self._try_lr(net, optimizer, start_lr, end_lr, num, beta)
