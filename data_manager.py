import os

import numpy as np
import SimpleITK as sitk
import threadpool
import multiprocessing
import skimage.transform

class DataManager:
    SKIP = 255
    TRUE = 1
    FALSE = 0

    def __init__(self, src_dir, result_dir, params, mode='train'):
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
        self.result_dir = result_dir
        self.mode = mode
        self.data_num = 0
        self.data_list = list()
        self.numpy_images = dict()
        self.numpy_gts = dict()
        self.pos = dict()
        self.func = dict()

        self.image_feed_type = np.float32
        self.gt_feed_type = np.uint8
        self.data_queue = None
        self.shapes = dict()

    def create_data_list(self):
        self.data_list = [
            path for path in os.listdir(self.src_dir)
            if self.image_name in os.listdir(os.path.join(self.src_dir, path))
        ]

    def load_data(self):
        self.create_data_list()
        self.run_load_thread()

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
                image, gt = self.data_augmentation(data, pos)

                image_list.append(image)
                gt_list.append(gt)

            self.data_queue.put((np.array(image_list), np.array(gt_list)))

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

    def _load_numpy_data(self, data):
        image_name = os.path.join(self.src_dir, data, self.image_name)
        gt_name = os.path.join(self.src_dir, data, self.gt_name)

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

    def _load_data_split(self, image):
        for k, i in enumerate(image.shape):
            if i % self.size:
                index_a = [slice(num) for num in image.shape]
                index_a[k] = slice(i // self.size * self.size)
                index_b = [slice(num) for num in image.shape]
                index_b[k] = slice(-self.size, i)
                image = np.concatenate((image[tuple(index_a)], image[tuple(index_b)]), k)

        for i in range(3):
            image = np.array(np.split(image, image.shape[-i] // self.size, -i))
        return image.reshape(-1, self.size, self.size)

    def run_load_thread(self):
        def load_callback(result):
            data, image, gt, pos_list, pad = result
            self.shapes[data] = image.shape
            self.numpy_images[data] = image
            self.numpy_gts[data] = gt
            self.pos[data] = pos_list
            self.func[data] = self.get_transfer(*pad, True)

        self.numpy_images.clear()
        self.numpy_gts.clear()
        self.shapes.clear()
        load_num = min(len(self.data_list), self.params['loadThreadNum'])
        if self.params['load_by_process']:
            pool = multiprocessing.Pool(load_num)
            result = pool.map_async(
                self._load_numpy_data, self.data_list,
                callback=lambda results: [load_callback(result) for result in results]
            )
            pool.close()
            pool.join()
        else:
            pool = threadpool.ThreadPool(load_num)
            reqs = threadpool.makeRequests(
                self._load_numpy_data, self.data_list,
                callback=lambda req, result: load_callback(result)
            )
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
        writer.SetFileName(os.path.join(self.src_dir, filename, self.result_name))
        writer.Execute(sitk_result)

    def reshape_to_sitk_image(self, data, name, type_='gt'):
        width, height, _ = self.shapes[name]

        if type_ == 'gt':
            dtype = self.gt_feed_type
        elif type_ == 'image':
            dtype = self.image_feed_type

        sitk_image = np.zeros(self.numpy_images[name].shape, dtype=dtype)
        sitk_image_count = np.zeros(self.numpy_images[name].shape, dtype=np.int)

        for num, image in enumerate(data):
            pos = self.func[name](num)
            sitk_image[pos] += image
            sitk_image_count[pos] += 1

        sitk_image = (sitk_image / sitk_image_count)[:width, :height, :]
        if type_ == 'gt':
            sitk_image = (sitk_image > self.params['reshape_threshold']).astype(dtype)
        return sitk_image

    def get_pad_size(self, shape):
        width, height, _ = shape
        pad_width_num = int(np.ceil((width - self.size) / self.stride)) + 1
        pad_height_num = int(np.ceil((height - self.size) / self.stride)) + 1
        pad_width = (pad_width_num - 1) * self.stride + self.size
        pad_height = (pad_height_num - 1) * self.stride + self.size
        return pad_width, pad_height, pad_width_num, pad_height_num

    def get_transfer(self, pad_width_num, pad_height_num, num_to_pos=True):
        def num_to_pos_func(num, offset_y=0, offset_x=0, slice_=True):
            i = num // (pad_width_num * pad_height_num)
            j = num // pad_height_num % pad_width_num * self.stride
            k = num % pad_height_num * self.stride
            if not slice_:
                return j+offset_y, k+offset_x, i
            return slice(j+offset_y, j+offset_y+self.size), slice(k+offset_x, k+offset_x+self.size), i

        def pos_to_num_func(pos):
            i, j, k = pos
            num = i * pad_width_num * pad_height_num + j / self.stride * pad_height_num + k / self.stride
            return int(num)

        if num_to_pos:
            return num_to_pos_func
        return pos_to_num_func

    def data_augmentation(self, data, pos):
        if self.params['random_offset']:
            offset_y = np.random.randint(self.stride)
            offset_x = np.random.randint(self.stride)
            # print('offset', offset_y, offset_x)
        else:
            offset_y = offset_x = 0

        j, k, i = self.func[data](pos, offset_y, offset_x, slice_=False)
        image = self.numpy_images[data][:,:,i]
        gt = self.numpy_gts[data][:,:,i]

        if self.params['random_scale']:
            scale_range = self.params['random_scale_range']
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

        if self.params['random_rotate']:
            rotate_range = self.params['random_rotate_range']
            angle = np.random.rand() * 2 * rotate_range - rotate_range
            # print('rotate', angle)
            image = skimage.transform.rotate(image, angle, center=(k, j))
            gt = skimage.transform.rotate(gt, angle, center=(k, j))

        image = image[j:j+self.size, k:k+self.size]
        gt = gt[j:j+self.size, k:k+self.size]
        gt = (gt > 0).astype(self.gt_feed_type)

        if self.params['random_flip']:
            if np.random.randint(2):
                image = np.flipud(image)
                gt = np.flipud(gt)
        return image, gt
