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

def load_image(filename) :
    return Image.open(filename)

def save_image(img, filename) :
    cv2.imwrite(filename, img)

#===============================================#
# function to load the model given a model name
#===============================================#
def load_model(model_name, pretrained):
    model = format('models.{}(pretrained={})'.format(model_name, pretrained))
    model = eval(model)

    return model

#=========================================================================#
# function to define the layer that will be used to extract the features
#=========================================================================#
def get_feat_vector_layer(model):
    model_name = model.__class__.__name__
    if model_name in ['AlexNet','VGG','ResNet']:
        layer = model.avgpool
    if model_name in ['SqueezeNet','DenseNet']:
        layer = model.features ###### verify this!
    return layer

#=========================================================================#
# function to define the size of the feature vector according to the model
#=========================================================================#
def get_feat_vector_size(model_name):
    if model_name == 'resnet18':
        size = 512
    else:
        size = 0
    return size

#=========================================================================#
# function to extract the feature vector from an image
#=========================================================================#
def extract_feature_vector(t_img, layer, model, feat_vect_size):
    # create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    feat_vect = torch.zeros((1, feat_vect_size, 1, 1))
    
    # define a function that will copy the output of a layer
    def copy_data(m, i, o):
        feat_vect.copy_(o.data)
    
    # attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    
    # run the model on our transformed image
    model(t_img)
    
    # detach our copy function from the layer
    h.remove()
    
    return feat_vect

#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Extracts the feature vectors from the images using a pre-trained CNN model')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_folder', type=str, help='Folder containing the images', required=True)
    required_named.add_argument('-o', '--output_file', type=str, help='Output file, numpy array (.npy)', required=True)
    parser.add_argument("--model_name", type=str, default='resnet18', choices=['alexnet','vgg11','vgg13','vgg16','vgg19',
                        'resnet18','resnet34','resnet50','resnet101','resnet152','squeezenet1_0','squeezenet1_1','densenet121',
                        'densenet161','densenet169','densenet201'], help="Pre-trained CNN model to extract the features")
    parser.add_argument("--torch_device", type=str, default='cuda:0', help="Device to execute the tests ('cpu','cuda:0','cuda:1',etc)")

    args = parser.parse_args()

    # read input parameters
    input_folder = args.input_folder
    output_file = args.output_file
    model_name = args.model_name
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

    # load the model
    model = load_model(model_name, "True")
    model.eval()

    # get the layer from where the features will be extracted and its size
    feat_vect_layer = get_feat_vector_layer(model)
    feat_vect_size = get_feat_vector_size(model_name)

    # define the transformations that will be applied to each image
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    # process each image
    time_start = time.time()
    feat_vects = np.zeros((len(files), feat_vect_size))
    for i, file in files_zip:
        print("Processing image {}/{} ...\r".format(i, len(files)), end="")
        # read image
        img = load_image(os.path.join(input_folder, file))

        # apply transformations
        t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

        # get the feature vectors
        feat_vect = extract_feature_vector(t_img, feat_vect_layer, model, feat_vect_size)
        feat_vects[i-1,:] = feat_vect.view(-1).numpy()
    print('\nDone! Time elapsed: {:.2f} seconds'.format(time.time()-time_start))
    
    np.save(output_file, feat_vects)
    print("Created file '{}' containing {} feature vectors with {} features each".format(output_file, feat_vects.shape[0], feat_vects.shape[1]))