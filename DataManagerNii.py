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

    def loadImages(self):
        self.sitkImages = dict()

        for f in tqdm(self.fileList):
            #self.sitkImages[f]=rescalFilt.Execute(sitk.Cast(sitk.ReadImage(join(join(self.srcFolder, f),'img.nii')),sitk.sitkFloat32))
            self.sitkImages[f] = sitk.Cast(sitk.ReadImage(
                join(join(self.srcFolder, f), 'img.nii')), sitk.sitkFloat32)
            


    def loadGT(self):
        self.sitkGT = dict()
        for f in tqdm(self.gtList):
            self.sitkGT[f] = sitk.Cast(sitk.ReadImage(
                join(join(self.srcFolder, f), 'label.nii')), sitk.sitkFloat32)
    
    def loadTrainingData(self):
        self.createImageFileList()
        self.createGTFileList()
        self.loadImages()
        self.loadGT()

    def loadTestData(self):
        self.createImageFileList()
        self.createGTFileList()
        self.loadImages()
        self.loadGT()
        self.numpyImages = self.getNumpyImages()

        '''self.numpyGTs= copy.deepcopy(self.numpyImages)
        for key in self.numpyGTs:
            self.numpyGTs[key][...]=0'''

        self.numpyGTs = self.getNumpyGT()

    def getNumpyData(self, dat, method):
        ret = dict()
        self.originalSizes = dict()
        for key in tqdm(dat):
            ret[key] = np.zeros([self.params['VolSize'][0], self.params['VolSize']
                                 [1], self.params['VolSize'][2]], dtype=np.float32)

            img = dat[key]
            ret[key] = sitk.GetArrayFromImage(img).astype(dtype=float)
            ret[key] = ret[key].astype(dtype=np.float32)
            self.originalSizes[key]=ret[key].shape
            #ret[key]=ret[key].astype(dtype=np.uint32)

        return ret

    def writeResultsFromNumpyLabel(self, result, key,original_image=False):
        if self.probabilityMap:
            result = result * 255
        else:
            pass
            # result = result>0.5
            # result = result.astype(np.uint8)
        result = np.transpose(result, [2, 1, 0])
        toWrite = sitk.GetImageFromArray(result)

        if original_image:
            toWrite = sitk.Cast(toWrite, sitk.sitkFloat32)
        else:
            toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)

        writer = sitk.ImageFileWriter()
        filename, ext = splitext(key)
        # print join(self.resultsDir, filename + '_result' + ext)
        writer.SetFileName(join(self.resultsDir, filename + '_result.nii.gz'))
        writer.Execute(toWrite)

class DataManagerNiiLazyLoad(DataManagerNii):
    def loadTestData(self):
        self.createImageFileList()
        #self.createGTFileList()
    
    def loadImgandLabel(self, f):
        img = sitk.Cast(sitk.ReadImage(
                join(join(self.srcFolder, f), 'img.nii.gz')), sitk.sitkFloat32)
        img = sitk.GetArrayFromImage(img).astype(dtype=np.float32)
        img = np.transpose(img, [2, 1, 0])

        label = np.zeros(img.shape)


        return img, label
    