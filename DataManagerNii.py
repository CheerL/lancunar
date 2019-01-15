import os

#import cv2
import numpy as np
import threadpool
import SimpleITK as sitk



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
        self.numpy_image_type = np.float32
        self.numpy_gt_type = np.int32
        self.data_queue = None
        self.pos_queue = None


    def create_data_list(self):
        self.data_list = [
            path for path in os.listdir(self.src_dir)
            if 'img.nii.gz' in os.listdir(os.path.join(self.src_dir, path))
        ]

    def load_data(self):
        self.create_data_list()
        self.run_load_thread()
        # for data in self.data_list:
        #     self.load_numpy_data(data)

    def run_train_processes(self):
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

        def data_load_process(pos_queue, data_queue):
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
                if gt.min() < 0:
                    print(each, gt.min())
                randomi = np.random.randint(4)
                image = np.rot90(image.copy(), randomi)
                gt = np.rot90(gt.copy(), randomi)

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
                target=data_load_process,
                args=(self.pos_queue, self.data_queue),
                daemon=True
            )
            load_process.start()

    def run_load_thread(self):
        pool = threadpool.ThreadPool(64)
        reqs = threadpool.makeRequests(self.load_numpy_data, [(data) for data in self.data_list])
        for req in reqs:
            pool.putRequest(req)
        pool.wait()

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
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_name)).transpose([2, 1, 0]).astype(self.numpy_image_type)
        image = image / image.max()
        if os.path.isfile(gt_name):
            gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_name)).transpose([2, 1, 0]).astype(self.numpy_gt_type)
        else:
            gt = np.zeros(image.shape, self.numpy_gt_type)
        if gt.min() < 0:
            print(data, gt.min())
        self.numpy_images[data] = image
        self.numpy_gts[data] = image
