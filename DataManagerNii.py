import copy
import math
import os
from os import listdir
from os.path import isfile, join, splitext

#import cv2
import numpy as np
import SimpleITK as sitk
import skimage.transform
from DataManager import DataManager
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool



class DataManagerNii(DataManager):
    def __init__(self, srcFolder, resultsDir, parameters, probabilityMap=False):
        self.num = 0
        self.resacle_filter = sitk.RescaleIntensityImageFilter()
        self.resacle_filter.SetOutputMaximum(1)
        self.resacle_filter.SetOutputMinimum(0)
        return super().__init__(srcFolder, resultsDir, parameters, probabilityMap=probabilityMap)

    def loadImages(self):
        self.sitkImages = dict()
        for path in tqdm(self.fileList):
            image_name = join(self.srcFolder, path, 'img.nii.gz')
            self.sitkImages[path] = self.resacle_filter.Execute(
                sitk.Cast(sitk.ReadImage(image_name), sitk.sitkFloat32)
            )

    def loadGT(self):
        self.sitkGTs = dict()
        for path in tqdm(self.gtList):
            gt_name = join(self.srcFolder, path, 'label.nii.gz')
            self.sitkGTs[path] = sitk.Cast(
                sitk.ReadImage(gt_name), sitk.sitkFloat32
            ) if isfile(gt_name) else None

    def loadData(self):
        self.createImageFileList()
        self.createGTFileList()
        self.loadImages()
        self.loadGT()
        self.numpyImages = self.getNumpyImages()
        self.numpyGTs = self.getNumpyGTs()
        assert len(self.numpyImages) == len(self.numpyGTs)
        self.padNumpyData()
        self.num = len(self.numpyImages)

    def getNumpyImages(self):
        numpy_images = {
            key: sitk.GetArrayFromImage(img).astype(dtype=np.float32).transpose([2, 1, 0])
            for key, img in tqdm(self.sitkImages.items())
        }
        return numpy_images

    def getNumpyGTs(self):
        numpyGTs = {
            key: (
                sitk.GetArrayFromImage(img).astype(dtype=np.float32).transpose([2, 1, 0])
                if img is not None else np.zeros(self.sitkImages[key].GetSize(), dtype=np.float32)
            ) for key, img in tqdm(self.sitkGTs.items())
        }
        return numpyGTs

    def writeResultsFromNumpyLabel(self, result, key,original_image=False):
        if self.probabilityMap:
            result = result * 255

        result = np.transpose(result, [2, 1, 0])
        toWrite = sitk.GetImageFromArray(result)

        if original_image:
            toWrite = sitk.Cast(toWrite, sitk.sitkFloat32)
        else:
            toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)

        writer = sitk.ImageFileWriter()
        filename, ext = splitext(key)
        writer.SetFileName(join(self.resultsDir, filename + '_result.nii.gz'))
        writer.Execute(toWrite)

    def padNumpyData(self):
        for key in self.numpyImages:
            image = self.numpyImages[key]
            gt = self.numpyGTs[key]

            padding = [max(j - i, 0) for i, j in zip(image.shape, self.params['VolSize'])]
            if any(padding):
                padding_size = tuple((0, pad) for pad in padding)
                self.numpyImages[key] = np.pad(image, padding_size, 'constant').astype(dtype=np.float32)
                self.numpyGTs[key] = np.pad(gt, padding_size, 'constant').astype(dtype=np.float32)


class DataManagerNiiLazyLoad(DataManagerNii):
    def loadData(self):
        self.createImageFileList()
        #self.createGTFileList()

    def loadImgandLabel(self, f):
        img = sitk.Cast(sitk.ReadImage(join(self.srcFolder, f, 'img.nii.gz')), sitk.sitkFloat32)
        img = sitk.GetArrayFromImage(img).astype(dtype=np.float32)
        img = np.transpose(img, [2, 1, 0])
        label = np.zeros(img.shape)
        return img, label
