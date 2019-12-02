#!/usr/bin/python

# author: Cesar Castelo, Julio Mendoza
# date: Nov 22, 2019
# description: Trains a ConvNet network using an image folder that contains train and val sets following the PyTorch convention.
# It uses cross validation to choose the best model
# adapted from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function, division

import subprocess
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn import metrics
import json
import csv
import pandas as pd
import glob
import argparse
import kfold
from sklearn.model_selection import StratifiedKFold
import collections

# data augmentation libraries
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable


import cv2

import train
import preprocess_data

# Reproducibility
'''
TEST IF ACC TIME CHANGES
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(389)
torch.manual_seed(389)
'''

#from albumentations.augmentations.transforms import ElasticTransform

def split_input(image, target_size):
    width, height = image.size
    patches = []
    for i in range(4):
        begin = i * 0.25
        end = begin + 0.25
        begin *= height
        end *= height
        patch = image.crop((0, begin, width, end))
        patch = patch.resize(target_size)
        patches.append(patch)
    return patches

def inference(model, image):
    # For transfer learning modes
    norm_mean, norm_stdev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    target_size=(224, 224)
    inference_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_stdev)
    ])
 
    patches = split_input(image, target_size)
    input_tensor = []

    for i in range(len(patches)):
        input_tensor.append(inference_transforms(patches[i]).float())
    input_tensor = torch.stack(input_tensor)

    input = Variable(input_tensor)
    input = input.to(device)
    output = model(input)
    _, preds = torch.max(output, 1)
    preds = preds.cpu().detach().numpy()
    return preds

def save_results(files, results, output_file):
    assert len(files) == len(results)
    results_array = [["Nombre de archivo", "0-25% de la imagen", "25-50% de la imagen", "50-75% de la imagen", "75-100% de la imagen"]]
    classes = ['Arcilla','Muy fino', 'Fino',
            'Grueso', 'Granulo o mayor', 'Medio',
            'Muy grueso', 'Limo', 'no'
            ]
    classes = sorted(classes)
    for i in range(len(files)):
        results_array.append([files[i], classes[results[i][0]], classes[results[i][1]], classes[results[i][2]], classes[results[i][3]]])

    #print(np.array(results_array))
    with open(output_file,'w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerows(results_array)

    '''
    Write xlsx file
    '''
    '''
    # xlsx does not work with deepo docker image
    writer = pd.ExcelWriter(output_file + '.xlsx', engine='xlsxwriter')
    df_results = pd.DataFrame(results_array)
    df_results.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    '''
    '''
    # openpyxls does not work with deepo docker image
    df_results = pd.DataFrame(results_array)
    df_results.to_excel(output_file + '.xlsx')
    '''

#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Trains a convnet with PyTorch models using Cross Validation strategy')


    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_dir', type=str, help='Path directory with the input images', required=True)
    required_named.add_argument('-m', '--model_file', type=str, help='Path to the model checkpoint', required=True)
    required_named.add_argument('-o', '--output_file', type=str, help='Path to the output file', required=True)

    parser.add_argument("--torch_device", type=str, default="cuda:0", help="Device to execute the tests ('cpu','cuda:0','cuda:1',etc)")

    args = parser.parse_args()

    # determine the torch device to be used (GPU device)
    if args.torch_device != "cpu" and not torch.cuda.is_available():
        print("CUDA is not available ... using CPU instead!")
        args.torch_device = "cpu"
    device = torch.device(args.torch_device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # create timer lists especified in the competition
    t_aux = []
    t2 = []

    # create additional timer lists
    t_inference = []

    # load input folder
    files = os.listdir(args.input_dir)
    files.sort()

    num_imgs = len(files)
    imgs = []
    for i in range(num_imgs):
        img = cv2.imread(os.path.join(args.input_dir, files[i]))
        imgs.append(img)

    # Load model
    t_load = time.time()
    model = torch.load(args.model_file)
    model.eval()
    print("[Adicional] Tiempo de carga del modelo a memoria:", time.time() - t_load)

    # First timer
    t1_inicio = time.time()

    # define the data transformations
    localizer = preprocess_data.ROILocalizer()

    #print("Tiempo para carga de componentes del modelo:", time.time() - t_loadm)
    results = []
    for i in range(num_imgs):
        # Keep time before preprocessing
        t_aux.append(time.time())

        # preprocesamiento
        well_id_if_specified = preprocess_data.get_well_id_from_filename(files[i])
        preproc_img = preprocess_data.process_img(imgs[i], well_id_if_specified, localizer)
        preproc_img = Image.fromarray(preproc_img)

        # Keep time after preprocessing
        t2.append(time.time())
        result = inference(model, preproc_img)
        results.append(result)

        # Keep time after evaluation
        t_inference.append(time.time())
    
    print("Tiempo total:", time.time() - t1_inicio)
    t_verf = 0
    for i in range(num_imgs):
        t_verf += t2[i] - t_aux[i]
    print("Tiempo de verificacion (preprocesamiento):", t_verf)
    t_eval = 0
    for i in range(num_imgs):
        t_eval += t_inference[i] - t2[i]
    print("[Adicional] Tiempo de total de evaluacion del modelo (inferencia):", t_eval)

    # Save results
    save_results(files, results, args.output_file)
