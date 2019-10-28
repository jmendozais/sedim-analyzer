#!/usr/bin/python

# author: Cesar Castelo
# date: Jun 24, 2019
# description: Trains a ConvNet network using an image folder that contains train and val sets following the PyTorch convention,
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

# data augmentation libraries
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.augmentations.transforms import ElasticTransform

#=============================#
# class for data augmentation
#=============================#
class SampleCropResize:
    def __init__(self, target_size):
        self.target_size = target_size
        
    def __call__(self, sample):
        image, label = sample
        width, height = image.size
        
        # Crop part
        begin = torch.rand((1,))
        begin *= .74
        end = begin + 0.25
        part_idx = ((begin + end)*2).type(torch.ByteTensor)
        begin = (begin * height).type(torch.ShortTensor).numpy()[0]
        end = (end * height).type(torch.ShortTensor).numpy()[0]
        image = image.crop((0, begin, width, end))
        image = image.resize(self.target_size)
        
        return image, label[part_idx.numpy()[0]]

#=============================#
# class for data augmentation
#=============================#    
class ElasticTransform:
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma
    
    def elastic_transform(self, image, alpha, sigma, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
        image = np.asarray(image)
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
        

        distored_image = map_coordinates(image, indices, order=1, mode='reflect')
        distored_image = distored_image.reshape(image.shape)
        return Image.fromarray(distored_image)

    def __call__(self, sample):
        return self.elastic_transform(sample, self.alpha, self.sigma)

#=============================#
# function to train the model
#=============================#
def train_model(model, phases, loss_function, optimizer, scheduler, n_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_kappa, best_acc = 0.0, 0.0

    # create json for results
    results_json = dict()
    for phase in phases:
        results_json[phase] = dict()
        results_json[phase]['loss'] = list()
        results_json[phase]['kappa'] = list()
        results_json[phase]['acc'] = list()

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch+1, n_epochs))

        # Each epoch has a training and validation phase
        for phase in subset_dirs:
            if phase == train_set_name:
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_labels = list()
            all_preds = list()

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == train_set_name):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_function(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == train_set_name:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels = all_labels + list(labels.cpu().data.detach().numpy())
                all_preds = all_preds + list(preds.cpu().detach().numpy())

            # print and save results
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_kappa = metrics.cohen_kappa_score(all_labels, all_preds)
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('- {} -> Loss: {:.4f}, Kappa: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_kappa, epoch_acc))

            results_json[phase]['loss'].append(epoch_loss)
            results_json[phase]['kappa'].append(epoch_kappa)
            results_json[phase]['acc'].append(epoch_acc.item())

            # deep copy the model
            if (phase == eval_set_name) and (epoch_kappa > best_kappa or (epoch_kappa == best_kappa and epoch_acc > best_acc)):
                best_kappa = epoch_kappa
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best from eval -> Kappa: {:4f}, Acc: {:4f}'.format(best_kappa, best_acc))

    results_json['best_eval_kappa'] = best_kappa
    results_json['best_eval_acc'] = best_acc.item()
    results_json['train_time'] = time_elapsed

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, results_json

#===============================================#
# function to load the model given a model name
#===============================================#
def load_model(model_name, pretrained):
    model = format('models.%s(pretrained=%s)' % (model_name, pretrained))
    model = eval(model)

    # aqui faltaria inicializar os pesos aleatoriamente

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

#=====================================================#
# function to define the loss function given its name
#=====================================================#
def define_loss_function(loss_func_name):
    loss_function = format('torch.nn.%s()' % (loss_func_name))
    loss_function = eval(loss_function)
    return loss_function

#===========================================================================#
# function to define the optimizer given its name and the chosen train mode
#===========================================================================#
def define_optimizer(optimizer_name, model, train_mode, learn_rate=0.001):
    # for fine-tuning and rnd-weights training modes we optimize all the parameters in the network
    if train_mode in ['fine-tuning','rnd-weights']:
        optm_params = 'model.parameters()'

    # for fixed-feats training mode we only optimize the parameters in the classification layer
    elif train_mode == 'fixed-feats':
        model_name = model.__class__.__name__
        if model_name in ['AlexNet','VGG','SqueezeNet','DenseNet']:
            optm_params = 'model.classifier.parameters()'
        if model_name == 'ResNet':
            optm_params = 'model.fc.parameters()'

    optimizer = format('torch.optim.%s(%s, lr=%f)' % (optimizer_name, optm_params, learn_rate))
    optimizer = eval(optimizer)
    return optimizer

#===============#
# main function
#===============#
if __name__ == "__main__":
    # get the size of the terminal window
    terminal_rows, terminal_cols = subprocess.check_output(['stty', 'size']).decode().split()
    terminal_cols = int(terminal_cols)

    # verify input parameters
    if(len(sys.argv) != 13):
        print("#"*terminal_cols)
        print("usage: train_convnet_pytorch.py <...>")
        print("[1] dataset_dir: Directory containing the train and eval image sets (following the PyTorch convention)")
        print("[2] subset_dirs: Directory names of the image subsets used to train the model (a two-elem list passed as string, e.g., ['train','eval'])")
        print("[3] model_name: Pre-trainned conv network to be used (the ones with * are the best for each category)")
        print("    'alexnet','vgg11','vgg13','vgg16','vgg19'(*),")
        print("    'resnet18','resnet34','resnet50','resnet101','resnet152'(*),")
        print("    'squeezenet1_0','squeezenet1_1'(*),")
        print("    'densenet121','densenet161'(*),'densenet169','densenet201'")
        print("[4] loss_func_name: Loss function, i.e., optimization criterion (the suggested one is marked with *):")
        print("    'L1Loss','MSELoss','CrossEntropyLoss'(*),'CTCLoss','NLLLoss','PoissonNLLLoss','KLDivLoss','BCELoss',")
        print("    'BCEWithLogitsLoss','MarginRankingLoss','HingeEmbeddingLoss','MultiLabelMarginLoss','SmoothL1Loss',")
        print("    'SoftMarginLoss','MultiLabelSoftMarginLoss','CosineEmbeddingLoss','MultiMarginLoss','TripletMarginLoss'")
        print("[5] optimizer_name: Optimization algorithm (the suggested one is marked with *):")
        print("    'Adadelta','Adagrad','Adam'(*),'SparseAdam','Adamax','ASGD','LBFGS','RMSprop','Rprop','SGD'")
        print("[6] train_mode: Training mode")
        print("    'rnd-weights': Load an arquitecture and initialize it with random weights")
        print("    'fine-tuning': Load pre-trained features and perform fine-tuning")
        print("    'fixed-feats': Load pre-trained features and freeze the feature layers")
        print("[7] batch_size: Batch size")
        print("[8] n_epochs: Number of epochs")
        print("[9] learn_rate: Learning rate")
        print("[10] torch_device: Device to execute the tests ('cpu','cuda:0','cuda:1',etc)")
        print("[11] output_dir_basename: Basename for the output folder")
        print("[12] output_suffix: Suffix to be added to the output files")
        print("#"*terminal_cols)
        sys.exit(-1)

    # read input parameters
    dataset_dir = sys.argv[1].rstrip('/')
    subset_dirs = eval(sys.argv[2])
    model_name = sys.argv[3]
    loss_func_name = sys.argv[4]
    optimizer_name = sys.argv[5]
    train_mode = sys.argv[6]
    batch_size = int(sys.argv[7])
    n_epochs = int(sys.argv[8])
    learn_rate = float(sys.argv[9])
    torch_device = sys.argv[10]
    output_dir_basename = sys.argv[11]
    output_suffix = sys.argv[12]

    # create the output folder
    output_dirname = output_dir_basename + '_' + model_name + '_' + loss_func_name + '_' + optimizer_name + '_' + train_mode
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    # get the names for the image subsets
    if len(subset_dirs) != 2:
        sys.exit("The number of image subsets in the subset_dirs list must be 2")
    train_set_name = subset_dirs[0]
    eval_set_name = subset_dirs[1]

    # create the data transformations
    target_size = [224, 224]
    padding = int(target_size[0]*0.1)
    data_transforms = {
        # Data augmentation and normalization for training
        train_set_name: transforms.Compose([
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1),
            transforms.Pad((padding, padding, padding, padding), padding_mode='reflect'),
            transforms.RandomAffine(degrees=10, 
                                    translate=(0.05, 0.05), 
                                    scale=(1.1, 1.25), 
                                    fillcolor=(0,0,0), resample=Image.BICUBIC),
            transforms.CenterCrop(target_size),
            transforms.RandomHorizontalFlip(),
            ElasticTransform(alpha=1200, sigma=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Just normalization for validation
        eval_set_name: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # load the image set
    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x]) for x in subset_dirs}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in subset_dirs}
    dataset_sizes = {x: len(image_datasets[x]) for x in subset_dirs}
    class_names = image_datasets[train_set_name].classes
    n_classes = len(class_names)

    # determine the torch device to be used
    if torch_device != "cpu" and not torch.cuda.is_available():
        print("CUDA is not available ... using CPU instead!")
        torch_device = "cpu"
    device = torch.device(torch_device)

    # print input parameters
    print("="*terminal_cols)
    print("Input parameters (train_convnet_pytorch.py script):")
    print("="*terminal_cols)
    print("dataset_dir: {}".format(dataset_dir))
    print("subset_dirs: {}".format(subset_dirs))
    print("n_samples_per_set:")
    for subset_name in subset_dirs:
        print("- {}: {}".format(subset_name, dataset_sizes[subset_name]))
    print("n_classes: {}".format(n_classes))
    print("model_name: {}".format(model_name))
    print("loss_func_name: {}".format(loss_func_name))
    print("optimizer_name: {}".format(optimizer_name))
    print("train_mode: {}".format(train_mode))
    print("batch_size: {}".format(batch_size))
    print("n_epochs: {}".format(n_epochs))
    print("learn_rate: {}".format(learn_rate))
    print("torch_device: {}".format(torch_device))
    print("output_dir_basename: {}".format(output_dir_basename))
    print("output_suffix: {}".format(output_suffix))
    print("="*terminal_cols)

    # load the model
    pretrained = "True" if train_mode in ['fine-tuning','fixed-feats'] else "False"
    model = load_model(model_name, pretrained)
    
    # fix the layers (only for fixed-feats training mode)
    if train_mode == 'fixed-feats':
        for param in model.parameters():
            param.requires_grad = False

    # update the classification layer to have the right number of classes
    model = update_classification_layer(model, n_classes)

    # copy the model to the chosen device
    model = model.to(device)

    # define the loss function
    loss_function = define_loss_function(loss_func_name)

    # define optimizer
    optimizer = define_optimizer(optimizer_name, model, train_mode, learn_rate=0.001)

    # decay learn_rate by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model, train_results = train_model(model, [train_set_name,eval_set_name], loss_function, optimizer, scheduler, n_epochs)

    # save results
    results_json = dict()
    results_json['dataset_dir'] = dataset_dir
    results_json['subset_dirs'] = subset_dirs
    results_json['n_samples_per_set'] = dict()
    for subset_name in subset_dirs:
        results_json['n_samples_per_set'][subset_name] = dataset_sizes[subset_name]
    results_json['n_classes'] = n_classes
    results_json['model_name'] = model_name
    results_json['loss_func_name'] = loss_func_name
    results_json['optimizer_name'] = optimizer_name
    results_json['train_mode'] = train_mode
    results_json['batch_size'] = batch_size
    results_json['n_epochs'] = n_epochs
    results_json['learn_rate'] = learn_rate
    results_json['output_dir_basename'] = output_dir_basename
    results_json['output_suffix'] = output_suffix
    results_json['train_results'] = train_results

    results_filename = os.path.join(output_dirname, "results_" + output_suffix + ".json")
    fp = open(results_filename, 'w')
    json.dump(results_json, fp, sort_keys=False, indent=4)

    # save the PyTorch model and the labels
    model_filename = os.path.join(output_dirname, "model_" + output_suffix + ".model")
    torch.save(model.state_dict(), model_filename)
    
    labels_filename = os.path.join(output_dirname, "labels_" + output_suffix + ".csv")
    df_labels = pd.DataFrame(list(image_datasets[train_set_name].classes))
    df_labels.to_csv(labels_filename, header=False, index=False)
