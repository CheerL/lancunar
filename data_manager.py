import os
from itertools import product

import numpy as np
import SimpleITK as sitk
import threadpool


class DataManager:
    def __init__(self, src_dir, result_dir, params):
        self.params = params
        self.src_dir = src_dir
        self.result_dir = result_dir

        self.data_num = 0
        self.data_list = list()
        self.numpy_images = dict()
        self.numpy_gts = dict()
        self.image_feed_type = np.float64
        self.gt_feed_type = np.int64
        self.data_queue = None

    def create_data_list(self):
        self.data_list = [
            path for path in os.listdir(self.src_dir)
            if 'img.nii.gz' in os.listdir(os.path.join(self.src_dir, path))
        ]

    def load_data(self):
        self.create_data_list()
        self.run_load_thread()

    def run_feed_thread(self):
        def data_shuffle_thread(pos_queue):
            true_pos = list()
            for data, gt in self.numpy_gts.items():
                for pos in np.where(gt.any((1, 2, 3))==True)[0]:
                    true_pos.append((data, pos))

            true_num = len(true_pos)
            false_num = int(true_num * self.params['MaxEmpty'])
            self.data_num = true_num + false_num

            while True:
                false_pos = list()
                false_num_list = np.random.rand(len(self.data_list))
                false_num_list = false_num_list * false_num / false_num_list.sum()

                for i, (data, gt) in enumerate(self.numpy_gts.items()):
                    for pos in np.random.choice(
                        np.where(gt.any((1, 2, 3))==False)[0],
                        int(round(false_num_list[i]))
                    ):
                        false_pos.append((data, pos))
                pos_list = true_pos + false_pos
                np.random.shuffle(pos_list)

                for pos in pos_list:
                    pos_queue.put(pos)

        def data_feed_thread(pos_queue):
            ''' the thread worker to prepare the training data'''
            data_size = (self.params['batchsize'], 1) + self.params['VolSize']
            while True:
                image_list = []
                gt_list = []
                for _ in range(self.params['batchsize']):
                    key, pos = pos_queue.get()
                    image = self.numpy_images[key][pos]
                    gt = self.numpy_gts[key][pos]
                    randomi = np.random.randint(4)
                    image = np.rot90(image, randomi)
                    gt = np.rot90(gt, randomi)

                    image_list.append(image)
                    gt_list.append(gt)

                self.data_queue.put((
                    np.array(image_list).reshape(data_size),
                    np.array(gt_list).reshape(data_size)
                ))

        self.data_queue = threadpool.Queue.Queue(self.params['dataQueueSize'])
        pos_queue = threadpool.Queue.Queue(
            self.params['feedThreadNum'] * self.params['dataQueueSize'] * self.params['batchsize']
        )

        pool = threadpool.ThreadPool(self.params['feedThreadNum'] + 1)
        feed_reqs = threadpool.makeRequests(
            data_feed_thread,
            [(pos_queue) for _ in range(self.params['feedThreadNum'])]
        )
        shuffle_reqs = threadpool.makeRequests(data_shuffle_thread, [(pos_queue)])

        for req in feed_reqs + shuffle_reqs:
            pool.putRequest(req)

    def run_load_thread(self):
        def data_split(image):
            for k, (i, j) in enumerate(zip(image.shape, self.params['VolSize'])):
                if i % j:
                    index_a = [slice(num) for num in image.shape]
                    index_a[k] = slice(i // j * j)
                    index_b = [slice(num) for num in image.shape]
                    index_b[k] = slice(-j, i)
                    image = np.concatenate((image[tuple(index_a)], image[tuple(index_b)]), k)

            for i in range(3):
                image = np.array(np.split(image, image.shape[-i] // self.params['VolSize'][-i], -i))
            return image.reshape(-1, *self.params['VolSize'])

        def load_numpy_data(data):
            image_name = os.path.join(self.src_dir, data, 'img.nii.gz')
            gt_name = os.path.join(self.src_dir, data, 'label.nii.gz')
            image = sitk.GetArrayFromImage(sitk.ReadImage(image_name)).transpose([2, 1, 0]).astype(self.image_feed_type)
            image = image / max(image.max(), 1)
            if os.path.isfile(gt_name):
                gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_name)).transpose([2, 1, 0]).astype(self.gt_feed_type)
            else:
                gt = np.zeros(image.shape, self.gt_feed_type)
            self.numpy_images[data] = data_split(image)
            self.numpy_gts[data] = data_split(gt)

        self.numpy_images.clear()
        self.numpy_gts.clear()
        pool = threadpool.ThreadPool(self.params['loadThreadNum'])
        reqs = threadpool.makeRequests(load_numpy_data, self.data_list)
        for req in reqs:
            pool.putRequest(req)
        pool.wait()

    def write_result(self, result, name):
        result = result.transpose([2, 1, 0])
        result = result > 0.5
        sitk_result = sitk.GetImageFromArray(result.astype(np.uint8))
        # sitk_result = sitk.Cast(sitk_result, sitk.sitkUInt8)
        writer = sitk.ImageFileWriter()
        filename, _ = os.path.splitext(name)
        writer.SetFileName(os.path.join(self.src_dir, filename, 'result.nii.gz'))
        writer.Execute(sitk_result)
