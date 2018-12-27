import sys
import os
import numpy as np
import SimpleITK as sitk
from os import listdir
from os.path import isfile, join, splitext
from scipy import ndimage
from scipy.ndimage import measurements

ground_truth_path = '/home/ljp/from_Dfwang/WML/testing'
results_path = './result/'
ground_truth_labels = dict()
results_labels = dict()


def load_labels():
    for folder in listdir(ground_truth_path):
        label = sitk.Cast(sitk.ReadImage(
            join(join(ground_truth_path, folder), 'label.nii')), sitk.sitkFloat32)
        # temp=sitk.GetArrayFromImage(label)
        # print temp.shape
        ground_truth_labels[folder] = np.transpose(sitk.GetArrayFromImage(
            label).astype(dtype=float), [2, 0, 1])
        # print ground_truth_labels[folder].shape
        label = sitk.Cast(sitk.ReadImage(
            join(results_path, folder + "_result.nii")), sitk.sitkFloat32)

        results_labels[folder] = np.transpose(sitk.GetArrayFromImage(
            label).astype(dtype=float), [2, 0, 1])
        results_labels[folder] = (
            results_labels[folder] ).astype(dtype=np.uint8)
        print(results_labels[folder].shape)


def dilation(img):
    s = ndimage.generate_binary_structure(3, 3)
    img = ndimage.binary_dilation(img, structure=s).astype(img.dtype)
    return img


def save_img(img, path):
    # print img.shape
    result = np.transpose(img, [1, 2, 0])
    # print result.shape
    toWrite = sitk.GetImageFromArray(result)
    toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(toWrite)


def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError(
            "Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def compute_metrix():
    dice_tot = 0.0
    for label_name in ground_truth_labels:

        GT_label = ground_truth_labels[label_name]
        result_label = results_labels[label_name]
        temp_dice = dice(GT_label, result_label)
        print("label_name: " + str(label_name), " ", temp_dice)
        dice_tot += temp_dice
    dice_avg = dice_tot / len(ground_truth_labels)
    return dice_avg


load_labels()
dice_avg = compute_metrix()
print("dice: ", dice_avg)
