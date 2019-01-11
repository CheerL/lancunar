import numpy as np
import SimpleITK as sitk
from os import listdir
from os.path import isfile, join, splitext
#import cv2
import copy
import math
import os
import skimage.transform


class DataManager(object):
    params = None
    srcFolder = None
    resultsDir = None

    fileList = None
    gtList = None

    sitkImages = None
    sitkGT = None
    numpyImages = None
    numpyGTs = None
    meanIntensityTrain = None
    probabilityMap = False
    originalSizes = False

    def __init__(self, srcFolder, resultsDir, parameters, probabilityMap=False):
        self.params = parameters
        self.srcFolder = srcFolder
        self.resultsDir = resultsDir
        self.probabilityMap = probabilityMap

    def createImageFileList(self):
        ''' find all directories containing images and put there name in the fileList'''
        self.fileList = list()
        for f in listdir(self.srcFolder):
            for file in listdir(join(self.srcFolder, f)):
                if 'img' in file:
                    self.fileList.append(f)
                    break


    def createGTFileList(self):
        '''find all directories containing labels and put there name in the fileList'''
        self.gtList = self.fileList

    def image_histogram_equalization(self, image, number_bins=256):
        # from
        # http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

        # get image histogram
        image_histogram, bins = np.histogram(
            image.flatten(), number_bins, normed=True)
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape), cdf

    def create_temp_images(self, img_name):
        # create the 2D image based on the sitk image
        img = sitk.GetArrayFromImage(
            self.sitkImages[img_name]).astype(dtype=float)
        shape = img.shape
        n = math.sqrt(shape[0])
        n = int(n)
        out_img = np.zeros([n * 512, n * 512])
        img = (img * 255).astype(np.uint8)
        for i in range(n):
            for j in range(n):
                out_img[i * 512:i * 512 + 512, j * 512:j *
                        512 + 512] = img[i * n + j, :, :]

        #cv2.imwrite(os.path.join('tempImg', img_name + '.png'), out_img)

    def loadImages(self):
        ''' load images from image.nii'''
        self.sitkImages = dict()
        for f in self.fileList:
            #self.sitkImages[f]=rescalFilt.Execute(sitk.Cast(sitk.ReadImage(join(join(self.srcFolder, f),'img.nii')),sitk.sitkFloat32))
            self.sitkImages[f] = sitk.Cast(sitk.ReadImage(
                join(join(self.srcFolder, f), 'img.nii')), sitk.sitkFloat32)


    def loadGT(self):
        ''' load labels from label.nii'''
        self.sitkGT = dict()
        for f in self.gtList:
            self.sitkGT[f] = sitk.Cast(sitk.ReadImage(
                join(join(self.srcFolder, f), 'label.nii')), sitk.sitkFloat32)

    def loadTrainingData(self):
        ''' load training data'''
        self.createImageFileList()
        self.createGTFileList()
        self.loadImages()
        self.loadGT()

    def loadTestData(self):
        ''' load testing data or validation data'''
        self.createImageFileList()
        self.createGTFileList()
        self.loadImages()
        self.loadGT()
        self.numpyImages = self.getNumpyImages()
        self.numpyGTs = self.getNumpyGT()
        
        #self.numpyGTs= copy.deepcopy(self.numpyImages)
        # for key in self.numpyGTs:
        #     self.numpyGTs[key][...]=0

        

    def getNumpyImages(self):
        dat = self.getNumpyData(self.sitkImages, sitk.sitkLinear)
        return dat

    def getNumpyGT(self):
        dat = self.getNumpyData(self.sitkGT, sitk.sitkLinear)
        return dat

    def getNumpyData(self, dat, method):
        ''' load numpy data from sitk data'''
        ret = dict()
        self.originalSizes = dict()
        for key in dat:
            ret[key] = np.zeros([self.params['VolSize'][0], self.params['VolSize']
                                 [1], self.params['VolSize'][2]], dtype=np.float32)

            img = dat[key]
            ret[key] = np.transpose(sitk.GetArrayFromImage(img).astype(dtype=float), [2, 1, 0])
            ret[key] = ret[key].astype(dtype=np.float32)
            
            # order =3 Bi-cubic, resize the whole image to the same size
            #skimage can only operate on the float(range: -1 to 1) and int image
            #reserve_range : bool, Whether to keep the original range of values. Otherwise, the input image is converted according to the conventions of img_as_float.
            self.originalSizes[key]=ret[key].shape
            ret[key]=ret[key].astype(dtype=np.uint32)
            #ret[key] = skimage.transform.resize(ret[key],self.params['WholeSize'],order=3, mode='reflect', preserve_range=True)
            ret[key]=ret[key].astype(dtype=np.float32)
            '''cv2.imshow("",ret[key][:,:,60])
            #cv2.waitKey(0)'''

        return ret

    def writeResultsFromNumpyLabel(self, result, key):
        ''' save the segmentation results to the result directory'''
        # toWrite=sitk.Image(img.GetSize()[0],img.GetSize()[1],img.GetSize()[2],sitk.sitkFloat32)

        # resize to the original size
        #result = skimage.transform.resize(result, self.originalSizes[key], order=3, mode='reflect', preserve_range=True)
        result = np.transpose(result, [2, 1, 0])

        if self.probabilityMap:
            result = result * 255
        else:
            result = result>0.5
            result = result.astype(np.uint8)
        toWrite = sitk.GetImageFromArray(result)

        toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)

        writer = sitk.ImageFileWriter()
        filename, ext = splitext(key)
        writer.SetFileName(join(self.resultsDir, filename + '_result.nii'))
        writer.Execute(toWrite)
