import os
from itertools import product

import numpy as np
import SimpleITK as sitk
import threadpool


class DataManagerNii:
    def __init__(self, src_dir, result_dir, params):
        self.params = params
        self.src_dir = src_dir
        self.result_dir = result_dir

        self.data_list = list()
        self.numpy_images = dict()
        self.numpy_gts = dict()
        self._numpy_images_max = dict()
        self.image_save_type = np.int32
        self.gt_save_type = np.int32
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
            height, width, depth = self.params['VolSize']
            stride_height, stride_width, stride_depth = self.params['TrainStride']

            pos_list = list()
            for key, image in self.numpy_images.items():
                whole_height, whole_width, whole_depth = image.shape
                for ystart, xstart, zstart in product(
                    range(0, whole_height-height, stride_height),
                    range(0, whole_width-width, stride_width),
                    range(0, whole_depth-depth, stride_depth)
                ):
                    pos_list.append((key, ystart, xstart, zstart))

            for _ in range(self.params['epoch']):
                np.random.shuffle(pos_list)
                for each in pos_list:
                    pos_queue.put(each)

        def data_feed_thread(pos_queue):
            ''' the thread worker to prepare the training data'''
            empty = 0
            height, width, depth = self.params['VolSize']

            while True:
                each = pos_queue.get()
                key, ystart, xstart, zstart = each
                slice_index = (
                    slice(ystart, ystart + height),
                    slice(xstart, xstart + width),
                    slice(zstart, zstart + depth)
                )
                image_max = self._numpy_images_max[key]
                image = self.numpy_images[key][slice_index] / image_max
                gt = self.numpy_gts[key][slice_index]

                if gt.any():
                    empty = 0
                elif empty < self.params['MaxEmpty']:
                    empty += 1
                else:
                    continue

                randomi = np.random.randint(4)
                image = np.rot90(image, randomi).astype(self.image_feed_type)
                gt = np.rot90(gt, randomi).astype(self.gt_feed_type)
                self.data_queue.put((image, gt))

        self.data_queue = threadpool.Queue.Queue(self.params['dataQueueSize'])
        pos_queue = threadpool.Queue.Queue(self.params['posQueueSize'])

        pool = threadpool.ThreadPool(self.params['feedThreadNum'] + 1)
        feed_reqs = threadpool.makeRequests(
            data_feed_thread,
            [(pos_queue) for _ in range(self.params['feedThreadNum'])]
        )
        shuffle_reqs = threadpool.makeRequests(data_shuffle_thread, [(pos_queue)])

        for req in feed_reqs + shuffle_reqs:
            pool.putRequest(req)

    def run_load_thread(self):
        def load_numpy_data(data):
            image_name = os.path.join(self.src_dir, data, 'img.nii.gz')
            gt_name = os.path.join(self.src_dir, data, 'label.nii.gz')
            image = sitk.GetArrayFromImage(sitk.ReadImage(image_name)).transpose([2, 1, 0]).astype(self.image_save_type)
            if os.path.isfile(gt_name):
                gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_name)).transpose([2, 1, 0]).astype(self.gt_save_type)
            else:
                gt = np.zeros(image.shape, self.gt_save_type)
            self.numpy_images[data] = image
            self.numpy_gts[data] = gt
            self._numpy_images_max[data] = image.max()

        # def kill_thread(tid):
        #     if not isinstance(tid, ctypes.c_longlong):
        #         tid = ctypes.c_longlong(tid)
        #     if not inspect.isclass(SystemExit):
        #         raise TypeError("Only types can be raised (not instances)")
        #     res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        #         tid, ctypes.py_object(SystemExit))
        #     if res == 0:
        #         raise ValueError("invalid thread id")
        #     elif res != 1:
        #         # """if it returns a number greater than one, you're in trouble,
        #         # and you should call it again with exc=NULL to revert the effect"""
        #         ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        #         raise SystemError("PyThreadState_SetAsyncExc failed")

        self.numpy_images.clear()
        self.numpy_gts.clear()
        self._numpy_images_max.clear()
        pool = threadpool.ThreadPool(self.params['loadThreadNum'])
        reqs = threadpool.makeRequests(load_numpy_data, [(data) for data in self.data_list])
        for req in reqs:
            pool.putRequest(req)
        pool.wait()

        # for worker in pool.workers:
        #     kill_thread(worker.ident)

    def write_result(self, result, name):
        result = result.transpose([2, 1, 0])
        result = sitk.GetImageFromArray(result, sitk.sitkUInt8)
        writer = sitk.ImageFileWriter()
        filename, _ = os.path.splitext(name)
        writer.SetFileName(os.path.join(self.result_dir, filename + '_result.nii.gz'))
        writer.Execute(result)

    def save_as_feed(self):
        for data in self.data_list:
            self.numpy_images[data] = self.numpy_images[data].astype(self.image_feed_type)
            self.numpy_gts[data] = self.numpy_gts[data].astype(self.gt_feed_type)
