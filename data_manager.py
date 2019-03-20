import os

import cv2
import numpy as np
import SimpleITK as sitk
import threadpool
import multiprocessing

# import visualizer
# vis = visualizer.Visualizer()

class Data:
    def __init__(self, image, gt, size, stride, sitk_image, bounding_box):
        assert image.shape == gt.shape
        self.image = image
        self.gt = gt
        self.shape = image.shape
        self.size = size
        self.stride = stride
        self.bounding_box = bounding_box
        self.ori_shape = sitk_image.GetSize()
        self.ori_origin = sitk_image.GetOrigin()
        self.ori_spacing = sitk_image.GetSpacing()
        self.ori_direction = sitk_image.GetDirection()
        self.ori_type = sitk_image.GetPixelIDValue()

        self.height_num = int(np.ceil((self.shape[0] - self.size) / self.stride)) + 1
        self.width_num = int(np.ceil((self.shape[1] - self.size) / self.stride)) + 1
        self.pos = np.zeros(self.shape[2] * self.width_num * self.height_num, np.uint8)
        self._offset = int(round(size / 6))
        self._mask = np.zeros((size, size))
        self._mask[self._offset:size-self._offset, self._offset:size-self._offset] = 1

    # def copy(self):
    #     copy_data = Data(self.image.copy(), self.gt.copy(), self.size, self.stride)
    #     copy_data.pos = self.pos.copy()
    #     copy_data.pad_shape = self.pad_shape
    #     copy_data.width_num = self.width_num
    #     copy_data.height_num = self.height_num
    #     return copy_data

    def num_to_pos_func(self, num, offset_y=0, offset_x=0, type_='image'):
        i = num // (self.width_num * self.height_num)               # depth
        j = num % self.height_num * self.stride                     # height
        k = num // self.height_num % self.width_num * self.stride   # width

        j = min(j+offset_y, self.shape[0] - self.size)
        k = min(k+offset_x, self.shape[1] - self.size)

        if type_ == 'num':
            return j, k, i
        elif type_ == 'slice':
            return slice(j, j+self.size), slice(k, k+self.size), i
        elif type_ == 'image':
            return self.image[j:j+self.size, k:k+self.size, i]
        elif type_ == 'gt':
            return self.gt[j:j+self.size, k:k+self.size, i] * self._mask

    def set_pos(self, skip_empty, skip_empty_rate):
        for num in range(self.pos.size):
            if self.num_to_pos_func(num, type_='gt').any():
                self.pos[num] = DataManager.TRUE
            elif skip_empty and (self.num_to_pos_func(num, type_='image') > 0).mean() < skip_empty_rate:
                self.pos[num] = DataManager.SKIP
            # else:
            #     self.pos[num] = DataManager.FALSE

    def data_augmentation(self, num, offset=False, flip=False, scale_range=0, rotate_range=0):
        offset_y = np.random.randint(self.stride) if offset else 0
        offset_x = np.random.randint(self.stride) if offset else 0

        j, k, i = self.num_to_pos_func(num, offset_y, offset_x, type_='num')
        image = self.image[:,:,i]
        gt = self.gt[:,:,i]

        rate = np.random.rand() * 2 * scale_range + 1 - scale_range if scale_range else 1.0
        angle = np.random.rand() * 2 * rotate_range - rotate_range if rotate_range else 0

        if angle or rate != 1.0:
            M = cv2.getRotationMatrix2D((k, j), angle, rate)
            shape = (image.shape[1], image.shape[0])
            image = cv2.warpAffine(image, M, shape)
            gt = cv2.warpAffine(gt, M, shape)

        image = image[j:j+self.size, k:k+self.size]
        gt = gt[j:j+self.size, k:k+self.size] * self._mask

        if flip:
            if np.random.randint(2):
                image = np.flipud(image)
                gt = np.flipud(gt)
        return image, gt > 0


class DataWithSeg(Data):
    def __init__(self, image, gt, seg, size, stride, sitk_image, bounding_box):
        self.seg = seg
        return super().__init__(image, gt, size, stride, sitk_image, bounding_box)

    def num_to_pos_func(self, num, offset_y=0, offset_x=0, type_='image'):
        if type_ == 'seg':
            j, k, i = super().num_to_pos_func(num, offset_y, offset_x, type_='num')
            return self.seg[j:j+self.size, k:k+self.size, i]
        else:
            return super().num_to_pos_func(num, offset_y, offset_x, type_)

    def set_pos(self):
        for num in range(self.pos.size):
            if self.num_to_pos_func(num, type_='gt').any():
                self.pos[num] = DataWithSegManager.TRUE
            elif not self.num_to_pos_func(num, type_='seg').any():
                self.pos[num] = DataWithSegManager.SKIP
            # else:
            #     self.pos[num] = DataManager.FALSE

    def data_augmentation(self, num, offset=False, flip=False, scale_range=0, rotate_range=0):
        offset_y = np.random.randint(self.stride) if offset else 0
        offset_x = np.random.randint(self.stride) if offset else 0
        j, k, i = self.num_to_pos_func(num, offset_y, offset_x, type_='num')
        rate = np.random.rand() * 2 * scale_range + 1 - scale_range if scale_range else 1.0
        angle = np.random.rand() * 2 * rotate_range - rotate_range if rotate_range else 0

        image = self.image[:,:,i]
        seg = self.seg[:,:,i]
        gt = self.gt[:,:,i]

        if angle or rate != 1.0:
            M = cv2.getRotationMatrix2D((k, j), angle, rate)
            shape = (image.shape[1], image.shape[0])
            image = cv2.warpAffine(image, M, shape)
            gt = cv2.warpAffine(gt, M, shape)
            seg = cv2.warpAffine(seg, M, shape)

        image = image[j:j+self.size, k:k+self.size]
        gt = gt[j:j+self.size, k:k+self.size] * self._mask
        seg = seg[j:j+self.size, k:k+self.size]
        # label = ((seg * gt).sum() + 0.001) / ((0.2 * seg * (1 - gt) + 0.8 * (1 - seg) * gt + seg * gt).sum() + 0.001) > 0.5
        label = gt.any()

        if flip:
            if np.random.randint(2):
                image = np.flipud(image)
                gt = np.flipud(gt)
                seg = np.flipud(seg)
        return image, gt > 0, seg > 0, label


class DataManager:
    SKIP = 255
    TRUE = 1
    FALSE = 0

    def __init__(self, src_dir, params, mode='train'):
        self.params = params
        self.size = params['size']
        if mode == 'test':
            self.stride = params['test_stride']
            self.batch_size = params['test_batch_size']
        else:
            self.stride = params['stride']
            self.batch_size = params['batch_size']
        self.image_name = params['image_name']
        self.gt_name = params['gt_name']
        self.result_name = params['result_name']
        self.src_dir = src_dir
        self.mode = mode
        self.data_num = 0
        self.data = dict()

        self.pool = None
        self.image_feed_type = np.float32
        self.gt_feed_type = np.uint8

    def get_bounding_box(self, sitk_data):
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(sitk.OtsuThreshold(sitk_data, 0, 255))
        return label_shape_filter.GetBoundingBox(255)

    def resample_sitk(self, sitk_data, bound):
        roi_data = sitk.RegionOfInterest(sitk_data, bound[3:], bound[:3])

        ref_spacing = (0.75, 0.75, roi_data.GetSpacing()[2])
        ref_size = [
            int(round((size-1)*space/ref_space+1))
            for ref_space, size, space
            in zip(ref_spacing, roi_data.GetSize(), roi_data.GetSpacing())
        ]

        ref = sitk.Image(ref_size, roi_data.GetPixelIDValue())
        ref.SetSpacing(ref_spacing)
        ref.SetOrigin(roi_data.GetOrigin())
        ref.SetDirection(roi_data.GetDirection())

        return sitk.Resample(roi_data, ref)

    def create_data_list(self):
        for path in os.listdir(self.src_dir):
            if self.image_name in os.listdir(os.path.join(self.src_dir, path)):
                self.data[path] = None

    def _load_numpy_data(self, name):
        image_name = os.path.join(self.src_dir, name, self.image_name)
        gt_name = os.path.join(self.src_dir, name, self.gt_name)

        sitk_image = sitk.ReadImage(image_name)
        bounding_box = self.get_bounding_box(sitk_image)
        image = sitk.GetArrayFromImage(
            self.resample_sitk(sitk_image, bounding_box)
        ).transpose([2, 1, 0]).astype(self.image_feed_type)
        image = image / max(image.max(), 1)
        if os.path.isfile(gt_name):
            sitk_gt = sitk.ReadImage(gt_name)
            gt = sitk.GetArrayFromImage(
                self.resample_sitk(sitk_gt, bounding_box)
            ).transpose([2, 1, 0]).astype(self.gt_feed_type)
        else:
            gt = np.zeros(image.shape, self.gt_feed_type)

        if sitk_image.GetDirection()[4] < 0:
            image = np.fliplr(image)
            gt = np.fliplr(gt)

        data = Data(image, gt, self.size, self.stride, sitk_image, bounding_box)
        # data.pad()
        data.set_pos(self.params['skip_empty'], self.params['skip_empty_rate'])
        return name, data

    def run_load_worker(self):
        def load_callback(name, data):
            self.data[name] = data

        if not self.data:
            self.create_data_list()

        load_num = min(len(self.data), self.params['load_worker_num'])
        pool = multiprocessing.Pool(load_num)
        pool.map_async(
            self._load_numpy_data, self.data.keys(),
            callback=lambda results: [load_callback(*result) for result in results]
        )
        pool.close()
        pool.join()

    def _feed_shuffle_worker(self, pos_queue):
        true_pos = list()

        for name in self.data.keys():
            data = self.data[name]
            for num in np.where(data.pos == self.TRUE)[0]:
                true_pos.append((name, num))

        true_num = len(true_pos)
        false_num = int(true_num * self.params['MaxEmpty'])
        self.data_num = true_num + false_num

        while True:
            false_pos = list()
            false_num_list = np.random.rand(len(self.data))
            false_num_list = np.round(false_num_list * false_num / false_num_list.sum()).astype(int)
            for i, name in zip(false_num_list, self.data.keys()):
                data = self.data[name]
                for num in np.random.choice(np.where(data.pos == self.FALSE)[0], i, replace=False):
                    false_pos.append((name, num))

            all_pos = true_pos + false_pos
            np.random.shuffle(all_pos)

            for pos in all_pos:
                pos_queue.put(pos)

    def _feed_data_worker(self, data_queue, pos_queue):
        ''' the thread worker to prepare the training data'''
        while True:
            name, num = pos_queue.get()
            data_queue.put(self.data[name].data_augmentation(
                num=num,
                offset=self.params['random_offset'],
                flip=self.params['random_flip'],
                scale_range=self.params['random_scale_range'],
                rotate_range=self.params['random_rotate_range'])
            )

    def run_feed_worker(self):
        data_queue = multiprocessing.Queue(self.params['data_queue_size'])
        pos_queue = multiprocessing.Queue(
            self.params['data_queue_size'] * self.batch_size
        )

        threadpool.threading.Thread(
            target=self._feed_shuffle_worker,
            args=(pos_queue,),
            daemon=True
        ).start()
        feed_processes = [
            multiprocessing.Process(
                target=self._feed_data_worker,
                args=(data_queue, pos_queue),
                daemon=True
            ).start()
            for _ in range(self.params['feed_worker_num'])
        ]
        return data_queue

    def write_result(self, result, name):
        sitk_data = self.reshape_to_sitk(result, name, 'gt')
        writer = sitk.ImageFileWriter()
        filename, _ = os.path.splitext(name)
        writer.SetFileName(os.path.join(self.src_dir, filename, self.result_name))
        writer.Execute(sitk_data)

    def reshape_to_sitk(self, result, name, type_='gt'):
        data = self.data[name]

        if type_ == 'gt':
            dtype = self.gt_feed_type
        elif type_ == 'image':
            dtype = self.image_feed_type

        np_data = np.zeros(data.shape, dtype=dtype)
        np_data_count = np.zeros(data.shape, dtype=np.int)

        for num, image in enumerate(result):
            pos = data.num_to_pos_func(num, type_='slice')
            np_data[pos] += image
            np_data_count[pos] += 1

        np_data_count[np_data_count==0] = 1
        np_data = (np_data / np_data_count)

        if data.ori_direction[4] < 0:
            np_data = np.fliplr(np_data)

        if type_ == 'gt':
            np_data = (np_data > self.params['reshape_threshold']).astype(dtype)
        np_data = np_data.transpose([2, 1, 0]).astype(np.uint8)
        sitk_data = sitk.GetImageFromArray(np_data)
        ref = sitk.Image(data.ori_shape, data.ori_type)
        ref.SetOrigin(data.ori_origin)
        ref.SetDirection(data.ori_direction)
        ref.SetSpacing(data.ori_spacing)
        return sitk.Resample(sitk_data, ref)


class DataWithSegManager(DataManager):
    def __init__(self, src_dir, params, mode='train'):
        super().__init__(src_dir, params, mode=mode)
        self.seg_feed_type = np.float32
        self.params['MaxEmpty'] = params['classify_MaxEmpty']
        self.seg_name = params['result_name']
        self.result_name = params['classify_result_name']

    def _load_numpy_data(self, name):
        image_name = os.path.join(self.src_dir, name, self.image_name)
        gt_name = os.path.join(self.src_dir, name, self.gt_name)
        seg_name = os.path.join(self.src_dir, name, self.seg_name)

        sitk_image = sitk.ReadImage(image_name)
        sitk_seg = sitk.ReadImage(seg_name)
        bounding_box = self.get_bounding_box(sitk_image)

        image = sitk.GetArrayFromImage(
            self.resample_sitk(sitk_image, bounding_box)
        ).transpose([2, 1, 0]).astype(self.image_feed_type)
        image = image / max(image.max(), 1)
        seg = sitk.GetArrayFromImage(
            self.resample_sitk(sitk_seg, bounding_box)
        ).transpose([2, 1, 0]).astype(self.seg_feed_type)
        if os.path.isfile(gt_name):
            sitk_gt = sitk.ReadImage(gt_name)
            gt = sitk.GetArrayFromImage(
                self.resample_sitk(sitk_gt, bounding_box)
            ).transpose([2, 1, 0]).astype(self.gt_feed_type)
        else:
            gt = np.zeros(image.shape, self.gt_feed_type)

        if sitk_image.GetDirection()[4] < 0:
            image = np.fliplr(image)
            gt = np.fliplr(gt)
            seg = np.fliplr(seg)

        data = DataWithSeg(image, gt, seg, self.size, self.stride, sitk_image, bounding_box)
        data.set_pos()
        return name, data
