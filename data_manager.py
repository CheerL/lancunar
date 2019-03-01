import os
from itertools import product

import numpy as np
import SimpleITK as sitk
import threadpool
import multiprocessing


class DataManager:
    def __init__(self, src_dir, result_dir, params, mode='train'):
        self.params = params
        self.size = params['size']
        self.stride = params['stride']
        self.batch_size = params['batchsize']
        self.src_dir = src_dir
        self.result_dir = result_dir
        self.mode = mode
        self.data_num = 0
        self.data_list = list()
        self.numpy_images = dict()
        self.numpy_gts = dict()
        self.pos = dict()
        self.image_feed_type = np.float16
        self.gt_feed_type = np.uint8
        self.data_queue = None
        self.shapes = dict()

    def create_data_list(self):
        self.data_list = [
            path for path in os.listdir(self.src_dir)
            if 'extimg.nii.gz' in os.listdir(os.path.join(self.src_dir, path))
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

                if self.params['random_rotate']:
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
        image_name = os.path.join(self.src_dir, data, 'extimg.nii.gz')
        gt_name = os.path.join(self.src_dir, data, 'label.nii.gz')
        sitk_image = sitk.ReadImage(image_name)
        image = sitk.GetArrayFromImage(sitk_image).transpose([2, 1, 0]).astype(self.image_feed_type)
        image = image / max(image.max(), 1)
        if os.path.isfile(gt_name):
            sitk_gt = sitk.ReadImage(gt_name)
            gt = sitk.GetArrayFromImage(sitk_gt).transpose([2, 1, 0]).astype(self.gt_feed_type)
        else:
            gt = np.zeros(image.shape, self.gt_feed_type)

        if sitk_image.GetDirection()[4] < 0:
            image = np.fliplr(image)
            gt = np.fliplr(gt)

        self.shapes[data] = image.shape
        self.numpy_images[data] = self._load_data_split(image)
        self.numpy_gts[data] = self._load_data_split(gt)
        if self.mode != 'test' and self.params['skip_empty']:
            pos = np.where((self.numpy_images[data] > 0).mean((1, 2)) > self.params['skip_empty_rate'])
            self.numpy_images[data] = self.numpy_images[data][pos]
            self.numpy_gts[data] = self.numpy_gts[data][pos]

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
    SKIP = 255
    TRUE = 1
    FALSE = 0

    def __init__(self, src_dir, result_dir, params, mode='train'):
        assert len(params['VolSize']) == 2
        self.func = dict()
        super().__init__(src_dir, result_dir, params, mode=mode)

    def get_pad_size(self, shape):
        width, height, _ = shape
        pad_width_num = int(np.ceil((width - self.size) / self.stride)) + 1
        pad_height_num = int(np.ceil((height - self.size) / self.stride)) + 1
        pad_width = (pad_width_num - 1) * self.stride + self.size
        pad_height = (pad_height_num - 1) * self.stride + self.size
        return pad_width, pad_height, pad_width_num, pad_height_num

    def get_transfer(self, pad_width_num, pad_height_num, num_to_pos=True):
        def num_to_pos_func(num):
            i = num // (pad_width_num * pad_height_num)
            j = num // pad_height_num % pad_width_num * self.stride
            k = num % pad_height_num * self.stride
            return slice(j, j+self.size), slice(k, k+self.size), i


        def pos_to_num_func(pos):
            i, j, k = pos
            num = i * pad_width_num * pad_height_num + j / self.stride * pad_height_num + k / self.stride
            return int(num)

        if num_to_pos:
            return num_to_pos_func
        else:
            return pos_to_num_func

    def _load_numpy_data(self, data):
        image_name = os.path.join(self.src_dir, data, 'extimg.nii.gz')
        gt_name = os.path.join(self.src_dir, data, 'label.nii.gz')

        sitk_image = sitk.ReadImage(image_name)
        image = sitk.GetArrayFromImage(sitk_image).transpose([2, 1, 0]).astype(self.image_feed_type)
        image = image / max(image.max(), 1)
        if os.path.isfile(gt_name):
            sitk_gt = sitk.ReadImage(gt_name)
            gt = sitk.GetArrayFromImage(sitk_gt).transpose([2, 1, 0]).astype(self.gt_feed_type)
        else:
            gt = np.zeros(image.shape, self.gt_feed_type)

        if sitk_image.GetDirection()[4] < 0:
            image = np.fliplr(image)
            gt = np.fliplr(gt)

        pad_width, pad_height, pad_width_num, pad_height_num = self.get_pad_size(image.shape)
        num_to_pos_func = self.get_transfer(pad_width_num, pad_height_num, True)
        pad = ((0, pad_width - image.shape[0]), (0, pad_height - image.shape[1]), (0, 0))
        image = np.pad(image, pad, mode='constant')
        gt = np.pad(gt, pad, mode='constant')

        pos_list = np.zeros(image.shape[2] * pad_width_num * pad_height_num, np.uint8)
        for pos in range(pos_list.size):
            if self.params['skip_empty'] and (image[num_to_pos_func(pos)] > 0).mean() < self.params['skip_empty_rate']:
                pos_list[pos] = self.SKIP
                continue

            if gt[num_to_pos_func(pos)].any():
                pos_list[pos] = self.TRUE
            # else:
            #     pos_list[pos] = self.FALSE
        return data, image, gt, pos_list, (pad_width_num, pad_height_num)

    def run_load_thread(self):
        self.numpy_images.clear()
        self.numpy_gts.clear()
        self.shapes.clear()
        # pool = threadpool.ThreadPool(self.params['loadThreadNum'])
        # reqs = threadpool.makeRequests(self._load_numpy_data, self.data_list)
        pool = multiprocessing.Pool(self.params['loadThreadNum'])
        result = pool.map(self._load_numpy_data, self.data_list)
        pool.close()
        pool.join()
        for data, image, gt, pos_list, pad in result:
            self.shapes[data] = image.shape
            self.numpy_images[data] = image
            self.numpy_gts[data] = gt
            self.pos[data] = pos_list
            self.func[data] = self.get_transfer(*pad, True)

    def _feed_data_shuffle_thread(self, pos_queue):
        true_pos = list()
        for data, pos_list in self.pos.items():
            for pos in np.where(pos_list == self.TRUE)[0]:
                true_pos.append((data, pos))

        true_num = len(true_pos)
        false_num = int(true_num * self.params['MaxEmpty'])
        self.data_num = true_num + false_num

        while True:
            false_pos = list()
            false_num_list = np.random.rand(len(self.data_list))
            false_num_list = np.round(false_num_list * false_num / false_num_list.sum()).astype(int)
            for i, (data, pos_list) in zip(false_num_list, self.pos.items()):
                for pos in np.random.choice(np.where(pos_list == self.FALSE)[0], i, replace=False):
                    false_pos.append((data, pos))
            all_pos_list = true_pos + false_pos
            np.random.shuffle(all_pos_list)
            for pos in all_pos_list:
                pos_queue.put(pos)

    def _feed_data_feed_thread(self, pos_queue):
        ''' the thread worker to prepare the training data'''
        while True:
            image_list = list()
            gt_list = list()

            for _ in range(self.batch_size):
                data, pos = pos_queue.get()
                image = self.numpy_images[data][self.func[data](pos)]
                gt = self.numpy_gts[data][self.func[data](pos)]
                if self.params['random_filp']:
                    if np.random.randint(2):
                        image = np.flipud(image)
                        gt = np.flipud(gt)

                image_list.append(image)
                gt_list.append(gt)

            self.data_queue.put((np.array(image_list), np.array(gt_list)))

    def reshape_to_sitk_image(self, data, name, type_='gt'):
        width, height, _ = self.shapes[name]

        if type_ == 'gt':
            dtype = self.gt_feed_type
        elif type_ == 'image':
            dtype = self.image_feed_type

        sitk_image = np.zeros(self.numpy_images[data].shape, dtype=dtype)
        sitk_image_count = np.zeros(self.numpy_images[data].shape, dtype=np.int)

        for num, image in enumerate(data):
            pos = self.func[data](num)
            sitk_image[pos] += image
            sitk_image_count[pos] += 1

        sitk_image = (sitk_image / sitk_image_count)[:width, :height, :]
        if type_ == 'gt':
            sitk_image = (sitk_image > self.params['reshape_threshold']).astype(dtype)
        return sitk_image
