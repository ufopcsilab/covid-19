#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: abr-03 of 2020
"""
import shutil

import cv2
import os
import pydicom

from covid19 import utils


def convert_dcm_to_png(data_type):

    inputdir = '/media/data/Profissional/Doc/Datasets/covid-19/rsna-pneumonia-detection-challenge/stage_2_{}_images/'.format(data_type)
    outdir = '/media/data/Profissional/Doc/Datasets/covid-19/COVID-Net-master/data/{}/'.format(data_type)
    # os.mkdir(outdir)

    test_list = [f for f in os.listdir(inputdir)]

    for f in test_list:   # remove "[:10]" to convert all images
        ds = pydicom.read_file(inputdir + f)  # read dicom image
        img = ds.pixel_array  # get image array
        cv2.imwrite(outdir + f.replace('.dcm', '.png'), img)  # write png image


def copy_missing_files(data_type):
    output_path = '/media/data/Profissional/Doc/Datasets/covid-19/COVID-Net-master/data/{}/'.format(data_type)
    train_dataset_path = [
        '/media/data/Profissional/Doc/Datasets/covid-19/covid-chestxray-dataset-master/images/',
        '/media/data/Profissional/Doc/Datasets/covid-19/rsna-pneumonia-detection-challenge/stage_2_{}_images/'.format(data_type)
    ]

    csv_content = utils.read_txt('/media/data/Profissional/Doc/Datasets/covid-19/COVID-Net-master/{}_COVIDx.txt'.format(data_type))

    _x_train_paths = []
    for c in csv_content:
        full_path = None
        img_path = c.split(' ')[-2]
        if not img_path.endswith('.dcm'):
            if os.path.exists(train_dataset_path[0] + img_path):
                full_path = train_dataset_path[0] + img_path
            elif os.path.exists(train_dataset_path[1] + img_path):
                full_path = train_dataset_path[1] + img_path
            # else:
            #     raise Exception('Impossible to find the file.....[{}]'.format(img_path))
            if full_path is not None:
                img = cv2.imread(full_path)
                cv2.imwrite(output_path + img_path, img)  # write png image


def check_if_all_files_exist(data_type):
    csv_content = utils.read_txt('/media/data/Profissional/Doc/Datasets/covid-19/COVID-Net-master/{}_COVIDx.txt'.format(data_type))
    imgs_path = '/media/data/Profissional/Doc/Datasets/covid-19/COVID-Net-master/data/{}/'.format(data_type)

    for c in csv_content:
        img_path = c.split(' ')[-2]
        if not os.path.exists(imgs_path + img_path):
            print('The following the message was not found [{}].'.format(img_path))
            shutil.move(imgs_path.replace('/test/', '/train/') + img_path, imgs_path + img_path)


if __name__ == '__main__':
    # convert_dcm_to_png('train')
    # convert_dcm_to_png('test')
    # copy_missing_files('train')
    # copy_missing_files('test')
    print('\n', '*' * 200)
    print('Train')
    check_if_all_files_exist('train')
    print('\n', '*' * 200)
    print('Test')
    check_if_all_files_exist('test')
