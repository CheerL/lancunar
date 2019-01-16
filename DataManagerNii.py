import os

#import cv2
import numpy as np
import threadpool
import SimpleITK as sitk
import ctypes
import inspect

from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Queue, Process
from itertools import product



class DataManagerNii:
    def __init__(self, src_dir, result_dir, params):
        self.params = params
        self.src_dir = src_dir
        self.result_dir = result_dir

        self.data_list = list()
        self.numpy_images = dict()
        self.numpy_gts = dict()
        self.image_save_type = np.int32
        self.gt_save_type = np.int32
        self.image_feed_type = np.float64
        self.gt_feed_type = np.int64
        self.data_queue = None
        self.pos_queue = None


    def create_data_list(self):
        self.data_list = [
            path for path in os.listdir(self.src_dir)
            if 'img.nii.gz' in os.listdir(os.path.join(self.src_dir, path))
        ]

    def load_data(self):
        self.create_data_list()
        # self.run_load_thread()
        for data in self.data_list:
            self.load_numpy_data(data)

    def run_feed_processes(self):
        def data_shuffle_process(pos_queue):
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

        def data_feed_process(pos_queue, data_queue):
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
                image = self.numpy_images[key][slice_index]
                gt = self.numpy_gts[key][slice_index]

                if gt.any():
                    empty = 0
                elif empty < self.params['MaxEmpty']:
                    empty += 1
                else:
                    continue

                randomi = np.random.randint(4)
                image = np.rot90(image.copy(), randomi).astype(self.image_feed_type)
                image = image / self.numpy_images[key].max()
                gt = np.rot90(gt.copy(), randomi).astype(self.gt_feed_type)
                data_queue.put((image, gt))

        self.data_queue = Queue(self.params['dataQueueSize'])
        self.pos_queue = Queue(self.params['posQueueSize'])
        shuffle_process = Process(
            target=data_shuffle_process,
            args=(self.pos_queue,),
            daemon=True
        )
        shuffle_process.start()

        for _ in range(self.params['nProc']):
            load_process = Process(
                target=data_feed_process,
                args=(self.pos_queue, self.data_queue),
                daemon=True
            )
            load_process.start()

    def run_load_thread(self):
        def kill_thread(tid):
            if not isinstance(tid, ctypes.c_longlong):
                tid = ctypes.c_longlong(tid)
            if not inspect.isclass(SystemExit):
                raise TypeError("Only types can be raised (not instances)")
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                tid, ctypes.py_object(SystemExit))
            if res == 0:
                raise ValueError("invalid thread id")
            elif res != 1:
                # """if it returns a number greater than one, you're in trouble,
                # and you should call it again with exc=NULL to revert the effect"""
                ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
                raise SystemError("PyThreadState_SetAsyncExc failed")

        pool = threadpool.ThreadPool(64)
        reqs = threadpool.makeRequests(self.load_numpy_data, [(data) for data in self.data_list])
        for req in reqs:
            pool.putRequest(req)
        pool.wait()

        for worker in pool.workers:
            kill_thread(worker.ident)

    def write_result(self, result, name):
        result = result.transpose([2, 1, 0])
        result = sitk.GetImageFromArray(result, sitk.sitkUInt8)
        writer = sitk.ImageFileWriter()
        filename, _ = os.path.splitext(name)
        writer.SetFileName(os.path.join(self.result_dir, filename + '_result.nii.gz'))
        writer.Execute(result)

    def load_numpy_data(self, data):
        image_name = os.path.join(self.src_dir, data, 'img.nii.gz')
        gt_name = os.path.join(self.src_dir, data, 'label.nii.gz')
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_name)).transpose([2, 1, 0]).astype(self.image_save_type)
        if os.path.isfile(gt_name):
            gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_name)).transpose([2, 1, 0]).astype(self.gt_save_type)
        else:
            gt = np.zeros(image.shape, self.gt_save_type)
        self.numpy_images[data] = image
        self.numpy_gts[data] = gt

    def save_as_feed(self):
        for data in self.data_list:
            self.numpy_images[data] = self.numpy_images[data].astype(self.image_feed_type)
            self.numpy_gts[data] = self.numpy_gts[data].astype(self.gt_feed_type)
