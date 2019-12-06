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

#===============================================#
# function to load the model given a model name
#===============================================#
def load_model(model_name, pretrained):
    model = format('models.%s(pretrained=%s)' % (model_name, pretrained))
    model = eval(model)

    return model

#=========================================================================#
# function to update the classification layer given the number of classes
#=========================================================================#
def update_classification_layer(model, n_classes):
    model_name = model.__class__.__name__
    if model_name in ['AlexNet','VGG']:
        n_feats = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(n_feats, n_classes)
    if model_name == 'ResNet':
        n_feats = model.fc.in_features
        model.fc = nn.Linear(n_feats, n_classes)
    if model_name == 'SqueezeNet':
        model.classifier[1] = nn.Conv2D(512, n_classes, kernel_size=(1,1), stride=(1,1))
    if model_name == 'DenseNet':
        n_feats = model.classifier.in_features
        model.classifier = nn.Linear(n_feats, n_classes)
    return model

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
    output = output.data.cpu().numpy()

    return preds, output

def save_results(files, results, output_file_csv):
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
    with open(output_file_csv,'w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerows(results_array)

    '''
    Write xlsx file
    '''
    '''
    # xlsx does not work with deepo docker image
    writer = pd.ExcelWriter(output_file_csv + '.xlsx', engine='xlsxwriter')
    df_results = pd.DataFrame(results_array)
    df_results.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    '''
    '''
    # openpyxls does not work with deepo docker image
    df_results = pd.DataFrame(results_array)
    df_results.to_excel(output_file_csv + '.xlsx')
    '''

def save_results_json(files, results, probabilities, t_eval, output_file_json):
    assert len(files) == len(results)
    results_dict = {}
    long_names = ["0-25% de la imagen", "25-50% de la imagen", "50-75% de la imagen", "75-100% de la imagen"]
    classes = sorted(['Arcilla','Muy fino', 'Fino', 'Grueso', 'Granulo o mayor', 'Medio', 'Muy grueso', 'Limo', 'no'])
    results_dict["classes"] = classes

    for i in range(len(files)):
        results_dict[files[i]] = {}
        for p in [1,2,3,4]:
            results_dict[files[i]]["patch_"+str(p)] = {}
            results_dict[files[i]]["patch_"+str(p)]["long_name"] = long_names[p-1]
            results_dict[files[i]]["patch_"+str(p)]["prediction"] = classes[results[i][p-1]]
            results_dict[files[i]]["patch_"+str(p)]["probabilities"] = [x/100.0 for x in probabilities[i][p-1].tolist()]
        results_dict[files[i]]["total_proc_time"] = t_eval

    result_file = open(output_file_json, 'w')
    json.dump(results_dict, result_file, sort_keys=True, indent=4)

#===============================================#
# Function to be called from outside the script
#===============================================#
def perform_inference(input_dir, model_file, output_file_csv, output_file_json="", torch_device="cuda:0", model_type="state-dict", model_name="densenet161", n_classes=9):
    # determine the torch device to be used (GPU device)
    if torch_device != "cpu" and not torch.cuda.is_available():
        print("CUDA is not available ... using CPU instead!")
        torch_device = "cpu"
    device = torch.device(torch_device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # create timer lists especified in the competition
    t_aux = []
    t2 = []

    # create additional timer lists
    t_inference = []

    # load input folder
    t_load = time.time()
    print("-> Cargando las imagenes a la memoria ...")
    files = os.listdir(input_dir)
    files.sort()

    num_imgs = len(files)
    imgs = []
    for i in range(num_imgs):
        img = cv2.imread(os.path.join(input_dir, files[i]))
        imgs.append(img)
    print("Tiempo:", time.time() - t_load)

    # Load model
    print("-> Cargando el modelo a la memoria ...")
    t_load = time.time()
    if model_type == "state-dict":
        model = load_model(model_name, pretrained='False')
        model = update_classification_layer(model, n_classes)
        model = model.to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
    elif model_type == "full-model":
        model = torch.load(model_file, map_location=device)
    model.eval()
    print("Tiempo:", time.time() - t_load)

    # First timer
    print("-> Classificando las imagenes ... ")
    t1_inicio = time.time()

    # define the data transformations
    localizer = preprocess_data.ROILocalizer()

    #print("Tiempo para carga de componentes del modelo:", time.time() - t_loadm)
    results, probabilities = [], []
    for i in range(num_imgs):
        # Keep time before preprocessing
        t_aux.append(time.time())

        # preprocesamiento
        well_id_if_specified = preprocess_data.get_well_id_from_filename(files[i])
        preproc_img = preprocess_data.process_img(imgs[i], well_id_if_specified, localizer)
        preproc_img = Image.fromarray(preproc_img)

        # Keep time after preprocessing
        t2.append(time.time())
        classes, probs = inference(model, preproc_img)
        results.append(classes)
        probabilities.append(probs)

        # Keep time after evaluation
        t_inference.append(time.time())
    
    t_verf = 0
    for i in range(num_imgs):
        t_verf += t2[i] - t_aux[i]
    t_eval = 0
    for i in range(num_imgs):
        t_eval += t_inference[i] - t2[i]
    print("Tiempo total: {} (pre-procesamiento: {} e inferencia: {})".format(time.time() - t1_inicio, t_verf, t_eval))

    # Save results
    save_results(files, results, output_file_csv)
    if output_file_json != "":
        save_results_json(files, results, probabilities, t_eval, output_file_json)

#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Trains a convnet with PyTorch models using Cross Validation strategy')


    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_dir', type=str, help='Path directory with the input images', required=True)
    required_named.add_argument('-m', '--model_file', type=str, help='Path to the model checkpoint (PyTorch state dict)', required=True)
    required_named.add_argument('-o', '--output_file_csv', type=str, help='Path to the output file', required=True)

    parser.add_argument("--output_file_json", type=str, default="", help="Path to the output JSON file (containing detailed info)")
    parser.add_argument("--torch_device", type=str, default="cuda:0", help="Device to execute the tests ('cpu','cuda:0','cuda:1',etc)")
    parser.add_argument("--model_type", type=str, default="state-dict", help="Format in which the model was saved", choices=["state-dict","full-model"])
    parser.add_argument("--model_name", type=str, default="densenet161", help="Name of the model used for training (use only when model_type='state-dict')", choices=['alexnet','vgg11','vgg13','vgg16','vgg19',
        'resnet18', 'resnet34','resnet50','resnet101','resnet152','squeezenet1_0','squeezenet1_1','densenet121','densenet161','densenet169','densenet201'])
    parser.add_argument("--n_classes", type=int, default=9, help="Number of classes present in the trained model (use only when model_type='state-dict')")

    args = parser.parse_args()
    input_dir = args.input_dir
    model_file = args.model_file
    output_file_csv = args.output_file_csv
    output_file_json = args.output_file_json
    torch_device = args.torch_device
    model_type = args.model_type
    model_name = args.model_name
    n_classes = args.n_classes

    # execute the main inference function
    perform_inference(input_dir, model_file, output_file_csv, output_file_json, torch_device, model_type, model_name, n_classes)
