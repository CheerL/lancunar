from __future__ import division, print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import mynet as CNN
from data_manager import DataManager
from logger import Logger
from visualizer import Visualizer
# from apex import amp
# amp_handle = amp.init()

class Model:
    ''' the network model for training, validation and testing '''
    train_manager = None
    val_manager = None
    test_manager = None
    min_loss = 9999999999
    min_loss_iou = 0
    max_iou = 0
    max_iou_loss = 0
    best_iteration_iou = 0
    best_iteration_loss = 0

    def __init__(self, params):
        self.params = params
        self.logger = Logger(__name__, self.params['dirLog'])
        self.vis = Visualizer(self.params['visname'])

    def all_predict(self, net, run_type='validation', silent=True, save=False):
        '''get the segmentation loss and iou of the validation data '''
        if run_type == 'validation':
            manager = self.val_manager
        elif run_type == 'testing':
            manager = self.test_manager

        loss_list = list()
        iou_list = list()
        for data in manager.data_list:
            result, loss, iou, gt_sum = self.predict(net, manager, data)
            loss_list.append(loss)
            iou_list.append(iou)

            if not silent:
                self.logger.info("{}: name: {} loss: {:.7f} iou: {:.5%} predict: {} gt: {}".format(
                    run_type, data, loss, iou, result.sum(), gt_sum
                ))

            if save:
                manager.write_result(result, data)

        return np.mean(loss_list), np.mean(iou_list)

    def predict(self, net, manager, data):
        ''' produce the segmentation result, one time one image'''
        net.eval()
        image = manager.numpy_images[data]
        gt = manager.numpy_gts[data]
        unskip_pos = np.where(manager.pos[data] != manager.SKIP)[0]
        num_to_pos_func = manager.func[data]
        size = manager.size
        batch_size = manager.batch_size
        result = np.zeros((manager.pos[data].size, size, size), dtype=manager.gt_feed_type)
        all_loss = list()
        all_iou = list()
        all_true_loss = list()
        all_true_iou = list()
        gt_sum = 0

        for start in range(0, unskip_pos.size, batch_size):
            end = min(start + batch_size, unskip_pos.size)
            pos = unskip_pos[start:end]
            image_block = np.array([image[num_to_pos_func(num)] for num in pos])
            gt_block = np.array([gt[num_to_pos_func(num)] for num in pos])

            data = torch.Tensor(image_block).cuda().float().view(-1, 1, size, size)
            labels = torch.Tensor(gt_block).cuda().long()
            logits = net(data)
            block_iou = net.iou(logits, labels).cpu().item()
            block_loss = net.loss(logits, labels).cpu().item()
            block_gt_sum = gt_block.sum()
            all_iou.append(block_iou)
            all_loss.append(block_loss)
            pred = net.get_pred(logits, size)
            result[pos] = pred.cpu().numpy()
            gt_sum += gt_block.sum()
            if block_gt_sum:
                all_true_iou.append(block_iou)
                all_true_loss.append(block_loss)
                # print(pred.sum(), block_gt_sum, block_iou, block_loss)

        iou = np.mean(all_iou)
        loss = np.mean(all_loss)
        print(np.mean(all_true_iou) if all_true_iou else 1, np.mean(all_true_loss) if all_true_loss else 0)
        return result, loss, iou, gt_sum

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

        self.train_manager = DataManager(
            self.params['dirTrain'],
            self.params['dirResult'],
            self.params['dataManager'],
            mode='train'
        )
        self.val_manager = DataManager(
            self.params['dirValidation'],
            self.params['dirResult'],
            self.params['dataManager'],
            mode='val'
        )
        self.train_manager.load_data()
        self.val_manager.load_data()

        self.train_manager.run_feed_thread()
        # create the network
        net = getattr(CNN, self.params['net'])(
            block_num=self.params['block_num'],
            loss_type=self.params['loss'],
            dropout=self.params['dropout'],
            ds_weight=self.params['ds_weight']
        )
        if self.params['snapshot'] > 0:
            self.logger.info("loading checkpoint " +
                             str(self.params['snapshot']))
            prefix_save = os.path.join(
                self.params['dirSnapshots'],
                self.params['tailSnapshots']
            )
            name = prefix_save + str(self.params['snapshot']) + '_' + "checkpoint.pth.tar"
            checkpoint = torch.load(name)
            net.load_state_dict(checkpoint['state_dict'])
            self.logger.info("loaded checkpoint " +
                             str(self.params['snapshot']))
        else:
            net.apply(self._weights_init)

        temp_loss = 0
        temp_iou = 0
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.params['baseLR'],
            momentum=0.9,
            weight_decay=self.params['weight_decay'],
        )

        net.train()
        net.cuda()
        t_max = max(
            self.params['min_tmax'],
            round(self.train_manager.data_num /
                  self.train_manager.batch_size * 8, -2)
        )
        # scheduler = CNN.RewarmCosineAnnealingLR(
        #     optimizer,
        #     T_max=t_max,
        #     eta_min=self.params['minLR']
        # )
        # scheduler = CNN.RewarmLongCosineAnnealingLR(
        #     optimizer,
        #     T_max=t_max,
        #     eta_min=self.params['minLR']
        # )
        # scheduler = CNN.SomeCosineAnnealingLR(
        #     optimizer,
        #     T_max=t_max,
        #     eta_min=self.params['minLR']
        # )
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.params['expgamma'])
        # scheduler = CNN.NoWorkLR(optimizer)
        scheduler = CNN.MultiStepLR(optimizer, self.params['step'], self.params['step_rate'])
        scheduler.last_epoch = snapshot - 1

        self.logger.info('Create scheduler max lr: {} min lr: {} Tmax: {}'.format(
            self.params['baseLR'], self.params['minLR'], t_max
        ))
        self.logger.info('Run {}'.format(net.net_name))
        size = self.train_manager.size

        for iteration in range(1, nr_iter+1):
            batch_image, batch_gt = self.train_manager.data_queue.get()
            data = torch.Tensor(batch_image).cuda().float().view(-1, 1, size, size)
            labels = torch.Tensor(batch_gt).cuda().long()
            logits = net(data)
            loss = net.loss(logits, labels)
            temp_loss += loss.cpu().item()
            temp_iou += net.iou(logits, labels).cpu().item()
            real_iteration = snapshot + iteration
            optimizer.zero_grad()
            loss.backward()
            scheduler.step()
            optimizer.step()

            if not iteration % train_interval:
                train_loss = temp_loss / train_interval
                train_iou = temp_iou / train_interval
                self.logger.info(
                    "training: iter: {} loss: {:.7f} iou: {:.5%}".format(
                        real_iteration,
                        train_loss,
                        train_iou
                    ))
                self.vis.plot_many({
                    'loss': train_loss,
                    'iou': train_iou,
                    'lr': optimizer.param_groups[0]['lr']
                }, x_start=snapshot + train_interval, x_step=train_interval)
                temp_loss = 0
                temp_iou = 0
                net.img(self.vis, data, labels, logits, self.train_manager.size)

            if not iteration % val_interval:
                # val_loss, val_iou = self.all_predict(
                #     net, silent=False)
                # if val_iou > self.max_iou:
                #     self.max_iou = val_iou
                #     self.max_iou_loss = val_loss
                #     self.best_iteration_iou = real_iteration

                # if val_loss < self.min_loss:
                #     self.min_loss = val_loss
                #     self.min_loss_iou = val_iou
                #     self.best_iteration_loss = real_iteration

                self.save_checkpoint({'iteration': real_iteration,
                                      'state_dict': net.state_dict(),
                                      'best_acc': self.best_iteration_loss == real_iteration},
                                     self.params['dirSnapshots'],
                                     self.params['tailSnapshots'])

                # self.logger.info(
                #     "validating: iteration: {} loss: {:.7f} iou: {:.5%}".format(
                #         real_iteration, val_loss, val_iou
                #     ))
                # self.logger.info(
                #     "validating: best_iou: {} loss: {:.7f} iou: {:.5%}".format(
                #         self.best_iteration_iou, self.max_iou_loss, self.max_iou
                #     ))
                # self.logger.info(
                #     "validating: best_loss: {} loss: {:.7f} iou: {:.5%}".format(
                #         self.best_iteration_loss, self.min_loss, self.min_loss_iou
                #     ))
                # self.vis.plot_many({
                #     'val_loss': val_loss,
                #     'val_iou': val_iou
                # }, x_start=snapshot + val_interval, x_step=val_interval)
                # net.train()

    def test(self):
        self.test_manager = DataManager(
            self.params['dirTest'],
            self.params['dirResult'],
            self.params['dataManager'],
            mode='test'
        )
        self.test_manager.load_data()

        net = getattr(CNN, self.params['net'])(
            block_num=self.params['block_num'],
            loss_type=self.params['loss'],
            dropout=self.params['dropout'],
            ds_weight=self.params['ds_weight']
        )
        prefix_save = os.path.join(
            self.params['dirSnapshots'],
            self.params['tailSnapshots']
        )
        name = prefix_save + str(self.params['snapshot']) + '_' + "checkpoint.pth.tar"
        checkpoint = torch.load(name)
        net.load_state_dict(checkpoint['state_dict'])
        net.cuda()
        loss, iou = self.all_predict(net, run_type='testing', silent=False, save=True)
        self.logger.info(
            'testing: loss: {:.7f} iou: {:.5%}'.format(
                loss, iou
            ))

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
        size = self.train_manager.size

        for iteration in range(1, num+1):
            update_lr(optimizer, lr)
            batch_image, batch_gt = self.train_manager.data_queue.get()
            data = torch.Tensor(batch_image).cuda().float().view(-1, 1, size, size)
            labels = torch.Tensor(batch_gt).cuda().long()

            logits = net(data)
            loss = net.loss(logits, labels)

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.cpu().item()
            smoothed_loss = avg_loss / (1 - beta ** iteration)
            # Stop if the loss is exploding
            if iteration > 1 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or iteration == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lr = np.log10(lr)
            log_lrs.append(log_lr)
            # Do the SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update the lr for the next step
            self.logger.info(
                'lr: {} log lr: {:.4f} loss: {:.7f} best loss: {:.7f}'.format(
                    lr, log_lr, smoothed_loss, best_loss
                ))
            lr *= factor
            if not iteration % 10 or iteration == num:
                plt.plot(log_lrs, losses)
                plt.savefig('find_lr.{}.{}.jpg'.format(start_lr, end_lr))

    def find_lr(self, start_lr=1e-7, end_lr=10.0, num=100, beta=0.9):
        self.train_manager = DataManager(
            self.params['dirTrain'],
            self.params['dirResult'],
            self.params['dataManager'],
            mode='train'
        )
        self.train_manager.load_data()
        self.train_manager.run_feed_thread()
        # create the network
        net = getattr(CNN, self.params['net'])(
            block_num=self.params['block_num'],
            loss_type=self.params['loss'],
            dropout=self.params['dropout'],
            ds_weight=self.params['ds_weight']
        )
        net.apply(self._weights_init)

        net.train()
        net.cuda()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.params['baseLR'],
            momentum=0.9,
            weight_decay=self.params['weight_decay'],
        )
        self._try_lr(net, optimizer, start_lr, end_lr, num, beta)
