import os

import time

import numpy as np
import SimpleITK as sitk
import threadpool
import multiprocessing
import skimage.transform
from functools import wraps

import itertools

def starmapstar(args):
    return list(itertools.starmap(args[0], args[1]))

class Data:
    def __init__(self, image, gt, size, stride):
        assert image.shape == gt.shape
        self.image = image
        self.gt = gt
        self.shape = image.shape
        self.size = size
        self.stride = stride

        self.pos = None
        self.pad_shape = None
        self.width_num = None
        self.height_num = None

    def pad(self):
        width, height, deepth = self.shape
        self.width_num = int(np.ceil((width - self.size) / self.stride)) + 1
        self.height_num = int(np.ceil((height - self.size) / self.stride)) + 1
        pad_width = (self.width_num - 1) * self.stride + self.size
        pad_height = (self.height_num - 1) * self.stride + self.size
        self.pad_shape = (pad_width, pad_height, deepth)

        pad = ((0, self.pad_shape[0] - self.shape[0]), (0, self.pad_shape[1] - self.shape[1]), (0, 0))
        self.image = np.pad(self.image, pad, mode='constant')

    def num_to_pos_func(self, num, offset_y=0, offset_x=0, type_='image'):
        i = num // (self.width_num * self.height_num)
        j = num // self.height_num % self.width_num * self.stride
        k = num % self.height_num * self.stride

        if type_ == 'num':
            return j+offset_y, k+offset_x, i
        elif type_ == 'slice':
            return slice(j+offset_y, j+offset_y+self.size), slice(k+offset_x, k+offset_x+self.size), i
        elif type_ == 'image':
            return self.image[j+offset_y:j+offset_y+self.size, k+offset_x:k+offset_x+self.size, i]
        elif type_ == 'gt':
            return self.gt[j+offset_y:j+offset_y+self.size, k+offset_x:k+offset_x+self.size, i]

    def pos_to_num_func(self, pos):
        i, j, k = pos
        num = i * self.width_num * self.height_num + j / self.stride * self.height_num + k / self.stride
        return int(num)

    def set_pos(self, skip_empty, skip_empty_rate):
        self.pos = np.zeros(self.shape[2] * self.width_num * self.height_num, np.uint8)
        for num in range(self.pos.size):
            if self.num_to_pos_func(num, type_='gt').any():
                self.pos[num] = DataManager.TRUE
            elif skip_empty and (self.num_to_pos_func(num, type_='image') > 0).mean() < skip_empty_rate:
                self.pos[num] = DataManager.SKIP
            # else:
            #     self.pos[num] = DataManager.FALSE

    def data_augmentation(self, num, params):
        if params['random_offset']:
            offset_y = np.random.randint(self.stride)
            offset_x = np.random.randint(self.stride)
            # print('offset', offset_y, offset_x)
        else:
            offset_y = offset_x = 0

        j, k, i = self.num_to_pos_func(num, offset_y, offset_x, type_='num')
        image = self.image[:,:,i]
        gt = self.gt[:,:,i]

        if params['random_scale']:
            scale_range = params['random_scale_range']
            y_rate = np.random.rand() * 2 * scale_range + 1 - scale_range
            x_rate = np.random.rand() * 2 * scale_range + 1 - scale_range
            # print('scale rate', y_rate, x_rate)
            image = skimage.transform.rescale(
                image, (y_rate, x_rate),
                multichannel=False, mode='constant', anti_aliasing=True
            )
            gt = skimage.transform.rescale(
                gt, (y_rate, x_rate),
                multichannel=False, mode='constant', anti_aliasing=True
            )
            j = int(j * y_rate)
            k = int(k * x_rate)

        if params['random_rotate']:
            rotate_range = params['random_rotate_range']
            angle = np.random.rand() * 2 * rotate_range - rotate_range
            # print('rotate', angle)
            image = skimage.transform.rotate(image, angle, center=(k, j))
            gt = skimage.transform.rotate(gt, angle, center=(k, j))

        image = image[j:j+self.size, k:k+self.size]
        gt = gt[j:j+self.size, k:k+self.size]

        if params['random_flip']:
            if np.random.randint(2):
                image = np.flipud(image)
                gt = np.flipud(gt)
        return image, gt > 0


class DataManager:
    SKIP = 255
    TRUE = 1
    FALSE = 0

    def __init__(self, src_dir, params, mode='train'):
        self.params = params
        self.size = params['size']
        self.stride = params['stride']
        if mode == 'test':
            self.batch_size = params['test_batch_size']
        else:
            self.batch_size = params['batch_size']
        self.image_name = params['image_name']
        self.gt_name = params['gt_name']
        self.result_name = params['result_name']
        self.src_dir = src_dir
        self.mode = mode
        self.data_num = 0
        if self.params['feed_by_process']:
            self.data = multiprocessing.Manager().dict()
        else:
            self.data = dict()

        self.pool = None
        self.image_feed_type = np.float32
        self.gt_feed_type = np.uint8

    def create_data_list(self):
        for path in os.listdir(self.src_dir):
            if self.image_name in os.listdir(os.path.join(self.src_dir, path)):
                self.data[path] = None

    def _load_numpy_data(self, name):
        image_name = os.path.join(self.src_dir, name, self.image_name)
        gt_name = os.path.join(self.src_dir, name, self.gt_name)

        sitk_image = sitk.ReadImage(image_name)
        image = sitk.GetArrayFromImage(sitk_image).transpose([2, 1, 0]).astype(self.image_feed_type)
        image = image / max(image.max(), 1)
        if os.path.isfile(gt_name):
            sitk_gt = sitk.ReadImage(gt_name)
            gt = sitk.GetArrayFromImage(sitk_gt).transpose([2, 1, 0]).astype(self.gt_feed_type)
        else:
            gt = np.zeros(image.shape, self.gt_feed_type)

        if sitk_image.GetDirection()[4] < 0 and self.mode == 'train':
            image = np.fliplr(image)
            gt = np.fliplr(gt)

        data = Data(image, gt, self.size, self.stride)
        data.pad()
        data.set_pos(self.params['skip_empty'], self.params['skip_empty_rate'])
        return name, data

    def run_load_worker(self):
        def load_callback(name, data):
            self.data[name] = data

        if not self.data:
            self.create_data_list()

        load_num = min(len(self.data), self.params['load_worker_num'])
        if self.params['load_by_process']:
            pool = multiprocessing.Pool(load_num)
            pool.map_async(
                self._load_numpy_data, self.data.keys(),
                callback=lambda results: [load_callback(*result) for result in results]
            )
            pool.close()
            pool.join()
        else:
            pool = threadpool.ThreadPool(load_num)
            reqs = threadpool.makeRequests(
                self._load_numpy_data, self.data.keys(),
                callback=lambda req, result: load_callback(*result)
            )
            for req in reqs:
                pool.putRequest(req)
            pool.wait()

    def _feed_shuffle_worker(self, pos_queue):
        print(3, pos_queue)
        true_pos = list()
        # print(self.data.keys())
        for name in self.data.keys():
            data = self.data[name]
            # print(name, type(data))
            for num in np.where(data.pos == self.TRUE)[0]:
                true_pos.append((name, num))

        true_num = len(true_pos)
        false_num = int(true_num * self.params['MaxEmpty'])
        self.data_num = true_num + false_num
        # print(true_pos[:10])
        while True:
            false_pos = list()
            false_num_list = np.random.rand(len(self.data))
            false_num_list = np.round(false_num_list * false_num / false_num_list.sum()).astype(int)
            for i, name in zip(false_num_list, self.data.keys()):
                data = self.data[name]
                for num in np.random.choice(np.where(data.pos == self.FALSE)[0], i, replace=False):
                    false_pos.append((name, num))
            # print(false_pos[:10])
            all_pos = true_pos + false_pos
            np.random.shuffle(all_pos)
            print(all_pos[:10])
            for pos in all_pos:
                # print(pos)
                pos_queue.put(pos)

    def _feed_data_worker(self, data_queue, pos_queue):
        ''' the thread worker to prepare the training data'''
        print(4, data_queue, pos_queue)
        while True:
            image_list = list()
            gt_list = list()

            for _ in range(self.batch_size):
                name, num = pos_queue.get(block=True)
                print(name, num)
                data = self.data[name]
                image, gt = data.data_augmentation(num, self.params)
                image_list.append(image.astype(self.image_feed_type))
                gt_list.append(gt.astype(self.gt_feed_type))
            print(data_queue.qsize(), pos_queue.qsize())
            data_queue.put((np.array(image_list), np.array(gt_list)))

    def get_feed_pool_and_manager(self):
        if self.params['feed_by_process']:
            return multiprocessing.Pool(self.params['feed_worker_num'] + 1), multiprocessing.Manager()
        else:
            return threadpool.ThreadPool(self.params['feed_worker_num'] + 1), None

    def run_feed_worker(self, pool, manager):
        if self.params['feed_by_process']:
            data_queue = manager.Queue(self.params['data_queue_size'])
            pos_queue = manager.Queue(
                self.params['data_queue_size'] * self.batch_size
            )
            print(0, data_queue, pos_queue)
            # pool = multiprocessing.Pool(self.params['feed_worker_num'] + 1)
            # pool.apply_async(self._feed_shuffle_worker, (pos_queue,))
            pool.map_async(self._feed_shuffle_worker, [pos_queue])
            pool.starmap_async(
                self._feed_data_worker, [(data_queue, pos_queue)] * self.params['feed_worker_num']
            )
            # pool._map_async(
            #     self._feed_data_worker, [(data_queue, pos_queue)] * self.params['feed_worker_num'], starmapstar
            # )
            # pool.close()
        else:
            data_queue = threadpool.Queue.Queue(self.params['data_queue_size'])
            pos_queue = threadpool.Queue.Queue(
                self.params['data_queue_size'] * self.batch_size
            )
            print(1, data_queue, pos_queue)

            # pool = threadpool.ThreadPool(self.params['feed_worker_num'] + 1)
            shuffle_reqs = threadpool.makeRequests(
                self._feed_shuffle_worker,
                [pos_queue]
            )
            feed_reqs = threadpool.makeRequests(
                self._feed_data_worker,
                [([data_queue, pos_queue], []) for _ in range(self.params['feed_worker_num'])]
            )


            for req in shuffle_reqs + feed_reqs:
                pool.putRequest(req)
        return data_queue

    def write_result(self, result, name):
        result = self.reshape_to_sitk_image(result, name, 'gt')
        result = result.transpose([2, 1, 0]).astype(np.uint8)
        sitk_result = sitk.GetImageFromArray(result)
        # sitk_result = sitk.Cast(sitk_result, sitk.sitkUInt8)
        writer = sitk.ImageFileWriter()
        filename, _ = os.path.splitext(name)
        writer.SetFileName(os.path.join(self.src_dir, filename, self.result_name))
        writer.Execute(sitk_result)

    def reshape_to_sitk_image(self, result, name, type_='gt'):
        data = self.data[name]
        width, height, _ = data.shape

        if type_ == 'gt':
            dtype = self.gt_feed_type
        elif type_ == 'image':
            dtype = self.image_feed_type

        sitk_image = np.zeros(data.pad_shape, dtype=dtype)
        sitk_image_count = np.zeros(data.pad_shape, dtype=np.int)

        for num, image in enumerate(result):
            pos = data.num_to_pos_func(num, type_='slice')
            sitk_image[pos] += image
            sitk_image_count[pos] += 1

        sitk_image = (sitk_image / sitk_image_count)[:width, :height]
        if type_ == 'gt':
            sitk_image = (sitk_image > self.params['reshape_threshold']).astype(dtype)
        return sitk_image
