import argparse
import numpy as np
from PIL import Image
import subprocess
import os, sys
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import csv
import pandas as pd

def load_image(filename) :
    return Image.open(filename)

def save_image(img, filename) :
    cv2.imwrite(filename, img)

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

#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Classify a image set using a saved CNN model')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_folder', type=str, help='Folder containing the images', required=True)
    required_named.add_argument('-m', '--trained_cnn_model', type=str, help='Trained CNN model', required=True)
    required_named.add_argument("--model_name", type=str, choices=['alexnet','vgg11','vgg13','vgg16','vgg19','resnet18',
        'resnet34','resnet50','resnet101','resnet152','squeezenet1_0','squeezenet1_1','densenet121','densenet161','densenet169',
        'densenet201'], help="Model name")
    required_named.add_argument('-l', '--class_labels', type=str, help='CSV containing the problem class labels', required=True)
    required_named.add_argument('-o', '--output_file', type=str, help='Output file containing the labels (.csv)', required=True)
    parser.add_argument("--torch_device", type=str, default='cuda:0', help="Device to execute the tests ('cpu','cuda:0','cuda:1',etc)")

    args = parser.parse_args()

    # read input parameters
    input_folder = args.input_folder
    trained_cnn_model = args.trained_cnn_model
    model_name = args.model_name
    class_labels = args.class_labels
    output_file = args.output_file
    torch_device = args.torch_device

    # read the files in the directory
    files = os.listdir(input_folder)
    files.sort()
    files_zip = zip(range(1, len(files)+1), files)

    # determine the torch device to be used
    if torch_device != "cpu" and not torch.cuda.is_available():
        print("CUDA is not available ... using CPU instead!")
        torch_device = "cpu"
    device = torch.device(torch_device)

    # read the problem class labels
    class_labels_list = pd.read_csv(class_labels, sep=",", header=None)
    class_labels_list = list(class_labels_list.iloc[:,0])

    # load the model
    model = load_model(model_name, "False")
    model = update_classification_layer(model, len(class_labels_list))
    model.load_state_dict(torch.load(trained_cnn_model))
    model.eval()

    # define the transformations that will be applied to each image
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    # process each image
    time_start = time.time()
    pred_labels = []
    for i, file in files_zip:
        print("Processing image {}/{} ...\r".format(i, len(files)), end="")
        # read image
        img = load_image(os.path.join(input_folder, file))

        # apply transformations
        t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

        # classify the image
        output = model(t_img)
        _, pred = torch.max(output, 1)
        pred = pred.data.cpu().numpy()[0]
        pred_labels.append([file, class_labels_list[pred]])
    print('\nDone! Time elapsed: {:.2f} seconds'.format(time.time()-time_start))

    # save the results
    with open(output_file, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        wr.writerows(pred_labels)
    print("Created file '{}' containing the predicted labels of the images".format(output_file))
    