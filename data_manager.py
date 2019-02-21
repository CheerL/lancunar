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
        self.batch_size = params['batchsize']
        self.data_list = list()
        self.numpy_images = dict()
        self.numpy_gts = dict()
        self.image_feed_type = np.float64
        self.gt_feed_type = np.int64
        self.data_queue = None
        self.shapes = dict()

    def create_data_list(self):
        self.data_list = [
            path for path in os.listdir(self.src_dir)
            if 'img.nii.gz' in os.listdir(os.path.join(self.src_dir, path))
        ]

    def load_data(self):
        self.create_data_list()
        self.run_load_thread()

    def _feed_data_shuffle_thread(self, pos_queue):
        dim = tuple(range(1, len(self.params['VolSize'])+1))
        true_pos = list()
        for data, gt in self.numpy_gts.items():
            for pos in np.where(gt.any(dim) == True)[0]:
                true_pos.append((data, pos))

        true_num = len(true_pos)
        false_num = int(true_num * self.params['MaxEmpty'])
        self.data_num = true_num + false_num
        while True:
            false_pos = list()
            false_num_list = np.random.rand(len(self.data_list))
            false_num_list = false_num_list * false_num / false_num_list.sum()

            for i, (data, gt) in enumerate(self.numpy_gts.items()):
                gt_false_list = np.where(gt.any(dim) == False)[0]
                for pos in np.random.choice(gt_false_list, int(round(false_num_list[i]))):
                    false_pos.append((data, pos))
            pos_list = true_pos + false_pos
            np.random.shuffle(pos_list)

            for pos in pos_list:
                pos_queue.put(pos)

    def _feed_data_feed_thread(self, pos_queue):
        ''' the thread worker to prepare the training data'''
        data_size = (self.batch_size, 1) + self.params['VolSize']
        while True:
            image_list = list()
            gt_list = list()
            for _ in range(self.batch_size):
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

    def run_feed_thread(self):
        self.data_queue = threadpool.Queue.Queue(self.params['dataQueueSize'])
        pos_queue = threadpool.Queue.Queue(
            self.params['feedThreadNum'] * self.params['dataQueueSize'] * self.batch_size
        )

        pool = threadpool.ThreadPool(self.params['feedThreadNum'] + 1)
        feed_reqs = threadpool.makeRequests(
            self._feed_data_feed_thread,
            [(pos_queue) for _ in range(self.params['feedThreadNum'])]
        )
        shuffle_reqs = threadpool.makeRequests(
            self._feed_data_shuffle_thread,
            [(pos_queue) for _ in range(self.params['suffleThreadNum'])]
        )

        for req in feed_reqs + shuffle_reqs:
            pool.putRequest(req)

    def _load_data_split(self, image):
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

    def _load_numpy_data(self, data):
        image_name = os.path.join(self.src_dir, data, 'img.nii.gz')
        gt_name = os.path.join(self.src_dir, data, 'label.nii.gz')
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_name)).transpose([2, 1, 0]).astype(self.image_feed_type)
        image = image / max(image.max(), 1)
        if os.path.isfile(gt_name):
            gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_name)).transpose([2, 1, 0]).astype(self.gt_feed_type)
        else:
            gt = np.zeros(image.shape, self.gt_feed_type)
        self.shapes[data] = image.shape
        self.numpy_images[data] = self._load_data_split(image)
        self.numpy_gts[data] = self._load_data_split(gt)

    def run_load_thread(self):
        self.numpy_images.clear()
        self.numpy_gts.clear()
        self.shapes.clear()
        pool = threadpool.ThreadPool(self.params['loadThreadNum'])
        reqs = threadpool.makeRequests(self._load_numpy_data, self.data_list)
        for req in reqs:
            pool.putRequest(req)
        pool.wait()

    def write_result(self, result, name):
        result = self.reshape_to_sitk_image(result, name, 'gt')
        result = result.transpose([2, 1, 0])
        result = result > 0.5
        sitk_result = sitk.GetImageFromArray(result.astype(np.uint8))
        # sitk_result = sitk.Cast(sitk_result, sitk.sitkUInt8)
        writer = sitk.ImageFileWriter()
        filename, _ = os.path.splitext(name)
        writer.SetFileName(os.path.join(self.src_dir, filename, 'result.nii.gz'))
        writer.Execute(sitk_result)

    def reshape_to_sitk_image(self, data, name, type_='gt'):
        image_width, image_height, image_depth = self.shapes[name]
        vol_width, vol_height, vol_depth = self.params['VolSize']

        width_step = image_width // vol_width
        depth_remain = image_depth % vol_depth
        depth_step = int(np.ceil(image_depth / vol_depth))

        if type_ == 'gt':
            dtype = self.gt_feed_type
        elif type_ == 'image':
            dtype = self.image_feed_type
        sitk_image = np.zeros((image_width, image_height, image_depth), dtype)

        for num, image in enumerate(data):
            ystart = (num % width_step) * vol_width
            xstart = (num // (width_step * depth_step)) * vol_height
            _zstart = num // width_step % depth_step
            zstart = _zstart * vol_depth if _zstart != depth_step - 1 else image_depth - vol_depth
            sitk_image[
                ystart:ystart+vol_width,
                xstart:xstart+vol_height,
                zstart:zstart+vol_depth
                ] += image

        if depth_remain:
            sitk_image[
                :, :, (image_depth - vol_depth):(image_depth - depth_remain)
            ] = sitk_image[
                :, :, (image_depth - vol_depth):(image_depth - depth_remain)
            ] / 2

        return sitk_image

class DataManager2D(DataManager):
    def __init__(self, src_dir, result_dir, params):
        assert len(params['VolSize']) == 2
        super().__init__(src_dir, result_dir, params)

    def _load_data_split(self, image):
        for k, (i, j) in enumerate(zip(image.shape[:2], self.params['VolSize'])):
            if i % j:
                index_a = [slice(num) for num in image.shape]
                index_a[k] = slice(i // j * j)
                index_b = [slice(num) for num in image.shape]
                index_b[k] = slice(-j, i)
                image = np.concatenate((image[tuple(index_a)], image[tuple(index_b)]), k)

        image = np.rollaxis(image, 2)
        for i in range(1, 3):
            image = np.array(np.split(image, image.shape[-i] // self.params['VolSize'][-i], -i))
        return image.reshape(-1, *self.params['VolSize'])

    def reshape_to_sitk_image(self, data, name, type_='gt'):
        image_width, image_height, image_depth = self.shapes[name]
        vol_width, vol_height = self.params['VolSize']

        width_step = int(np.ceil(image_width / vol_width))
        width_remain = image_width % vol_width
        height_step = int(np.ceil(image_height / vol_height))
        height_remain = image_height % vol_height
        depth_step = image_depth
        if type_ == 'gt':
            dtype = self.gt_feed_type
        elif type_ == 'image':
            dtype = self.image_feed_type
        sitk_image = np.zeros((image_width, image_height, image_depth), dtype)

        for num, image in enumerate(data):
            z = num % depth_step
            _ystart = num // (height_step * depth_step)
            ystart = _ystart * vol_width if _ystart != width_step - 1 else image_width - vol_width
            _xstart = num // depth_step % height_step
            xstart = _xstart * vol_height if _xstart != height_step - 1 else image_height - vol_height
            sitk_image[ystart:ystart+vol_width, xstart:xstart+vol_height, z] += image

        if width_remain:
            sitk_image[
                (image_width - vol_width):(image_width - width_remain), :, :
            ] = sitk_image[
                (image_width - vol_width):(image_width - width_remain), :, :
            ] / 2

        if height_remain:
            sitk_image[
                :, (image_height - vol_height):(image_height - height_remain), :
            ] = sitk_image[
                :, (image_height - vol_height):(image_height - height_remain), :
            ] / 2

        return sitk_image
