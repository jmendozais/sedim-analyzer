#!/usr/bin/python

# author: Cesar Castelo
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
from sklearn.model_selection import StratifiedKFold
import collections

# data augmentation libraries
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from albumentations.augmentations.transforms import ElasticTransform

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

#=======================================#
# functions to perform cross-validation
#=======================================#
# distribution of the folds
def partitions(number, k):
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[0:(number % k)] += 1
    return n_partitions

# indices of the set eval
def get_indices(n_splits = 3, subjects = 145, frames = 20):
    l = partitions(subjects, n_splits)
    fold_sizes = l * frames
    indices = np.arange(subjects * frames).astype(int)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop =  current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])

# generates folds for cross validation
def k_folds(n_splits = 3, subjects = 145, frames = 20):
    indices = np.arange(subjects * frames).astype(int)
    for eval_idx in get_indices(n_splits, subjects, frames):
        train_idx = np.setdiff1d(indices, eval_idx)
        yield train_idx, eval_idx

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
        for phase in img_subset_names:
            if phase == train_set_name:
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_labels = list()
            all_preds = list()

            # Iterate over data.
            b = 1
            for inputs, labels in dataloaders[phase]:
                print("- [{}] batch: {}/{}\r".format(phase, b, len(dataloaders[phase])), end="")
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
                b += 1

                # print("outputs:", outputs)
                # print("preds:", list(preds.cpu().data.detach().numpy()))
                # print("labels:", list(labels.cpu().data.detach().numpy()))
                # print("preds:", collections.Counter(list(preds.cpu().data.detach().numpy())).keys())
                # print("labels:", collections.Counter(list(labels.cpu().data.detach().numpy())).keys())

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels = all_labels + list(labels.cpu().data.detach().numpy())
                all_preds = all_preds + list(preds.cpu().detach().numpy())

            # print and save results
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_kappa = metrics.cohen_kappa_score(all_labels, all_preds)
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('\n- [{}] Loss: {:.4f}, Kappa: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_kappa, epoch_acc))

            results_json[phase]['loss'].append(epoch_loss)
            results_json[phase]['kappa'].append(epoch_kappa)
            results_json[phase]['acc'].append(epoch_acc.item())

            # deep copy the model
            # if (phase == eval_set_name) and (epoch_kappa > best_kappa or (epoch_kappa == best_kappa and epoch_acc > best_acc)):
            if (phase == eval_set_name) and epoch_acc > best_acc:
                best_kappa = epoch_kappa
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
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
    # for transfer-learning and rnd-weights training modes we optimize all the parameters in the network
    if train_mode in ['transfer-learning','rnd-weights','rnd-weights-full-img-size']:
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

#=====================================================#
# function to define the data transformations
#=====================================================#
def define_data_transformations(data_augmentation, train_mode, img_dir_df):
    # define the target sizes for the images
    if train_mode in ['transfer-learning','fixed-feats','rnd-weights']:
        target_size = [224, 224]
    elif train_mode == 'rnd-weights-full-img-size':
        # get the mean size of the images
        sizes = []
        for img in img_dir_df:
            img = Image.open(img)
            sizes.append((img.shape[0]+img.shape[1])/2.0)
        target_size = np.mean(sizes)
        target_size = [target_size, target_size]

    # define the mean and stdev values for normalization
    if train_mode in ['transfer-learning','fixed-feats']:
        norm_mean, norm_stdev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif train_mode in ['rnd-weights','rnd-weights-full-img-size']:
        norm_mean, norm_stdev = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    # define the transformations
    if data_augmentation == 'simple':
        data_transforms = {
            # Data augmentation and normalization for training
            train_set_name: transforms.Compose([
                transforms.RandomResizedCrop(target_size[0]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_stdev)
            ]),
            # Just normalization for validation
            eval_set_name: transforms.Compose([
                transforms.Resize(int(target_size[0]*1.14)),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_stdev)
            ]),
        }
    elif data_augmentation == 'advanced':
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
                transforms.Normalize(norm_mean, norm_stdev)
            ]),
            # Just normalization for validation
            eval_set_name: transforms.Compose([
                transforms.Resize(int((target_size[0]*1.14))),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_stdev)
            ]),
        }
    
    return data_transforms


class MyDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        self.image_paths = list(data_frame.iloc[:,0])
        factor = pd.factorize(data_frame.iloc[:,1])
        self.targets = factor[0]
        self.labels = factor[1]
        self.transform = transform
        
    def __getitem__(self, index):
        # Load actual image here
        x = Image.open(self.image_paths[index])
        if self.transform:
            x = self.transform(x)
        y = self.targets[index]
        
        return x, y
    
    def __len__(self):
        return len(self.image_paths)


#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Trains a convnet with PyTorch models using Cross Validation strategy')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_dataset', type=str, help='CSV containing the image set and labels (two columns)',
        required=True)
    required_named.add_argument('-o', '--output_dir_basename', type=str, help='Basename for the output folder', required=True)
    parser.add_argument("--model_name", type=str, default="alexnet", help="Pre-trainned conv network to be used", choices=['alexnet','vgg11','vgg13','vgg16','vgg19',
        'resnet18', 'resnet34','resnet50','resnet101','resnet152','squeezenet1_0','squeezenet1_1','densenet121','densenet161','densenet169','densenet201'])
    parser.add_argument("--loss_func_name", type=str, default="CrossEntropyLoss", help="Loss function (optimization criterion)", choices=['L1Loss','MSELoss',
        'CrossEntropyLoss','CTCLoss','NLLLoss','PoissonNLLLoss','KLDivLoss','BCELoss','BCEWithLogitsLoss','MarginRankingLoss','HingeEmbeddingLoss',
        'MultiLabelMarginLoss','SmoothL1Loss','SoftMarginLoss','MultiLabelSoftMarginLoss','CosineEmbeddingLoss','MultiMarginLoss','TripletMarginLoss'])
    parser.add_argument("--optimizer_name", type=str, default="Adam", help="Optimization algorithm", choices=['Adadelta','Adagrad','Adam','SparseAdam','Adamax',
        'ASGD','LBFGS','RMSprop','Rprop','SGD'])
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds to perform cross-validation")
    parser.add_argument("--train_mode", type=str, default="transfer-learning", help="Training mode", choices=['transfer-learning','fixed-feats','rnd-weights',
        'rnd-weights-full-img-size'])
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learn_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data_augmentation", type=str, default="simple", help="Data augmentation strategy to be used", choices=['simple','advanced'])
    parser.add_argument("--torch_device", type=str, default="cuda:0", help="Device to execute the tests ('cpu','cuda:0','cuda:1',etc)")
    parser.add_argument("--output_suffix", type=str, default="convnet", help="Suffix to be added to the output files")
    args = parser.parse_args()

    # read input parameters
    input_dataset = args.input_dataset
    output_dir_basename = args.output_dir_basename.rstrip('/')
    model_name = args.model_name
    loss_func_name = args.loss_func_name
    optimizer_name = args.optimizer_name
    n_folds = args.n_folds
    train_mode = args.train_mode
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learn_rate = args.learn_rate
    data_augmentation = args.data_augmentation
    torch_device = args.torch_device
    output_suffix = args.output_suffix

    # print input parameters
    print("Input parameters:")
    print("input_dataset: {}".format(input_dataset))
    print("output_dir_basename: {}".format(output_dir_basename))
    print("model_name: {}".format(model_name))
    print("loss_func_name: {}".format(loss_func_name))
    print("optimizer_name: {}".format(optimizer_name))
    print("n_folds: {}".format(n_folds))
    print("train_mode: {}".format(train_mode))
    print("batch_size: {}".format(batch_size))
    print("n_epochs: {}".format(n_epochs))
    print("learn_rate: {}".format(learn_rate))
    print("data_augmentation: {}".format(data_augmentation))
    print("torch_device: {}".format(torch_device))
    print("output_suffix: {}".format(output_suffix))
    print()

    # create the output folder
    output_dirname = output_dir_basename + '_' + model_name + '_' + loss_func_name + '_' + optimizer_name + '_' + train_mode
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    # define the names for the image subsets
    img_subset_names = ["train", "eval"]
    train_set_name = img_subset_names[0]
    eval_set_name = img_subset_names[1]

    # determine the torch device to be used
    if torch_device != "cpu" and not torch.cuda.is_available():
        print("CUDA is not available ... using CPU instead!")
        torch_device = "cpu"
    device = torch.device(torch_device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # define the data transformations
    data_frame = pd.read_csv(input_dataset, delimiter=',', header=0)
    data_transforms = define_data_transformations(data_augmentation, train_mode, data_frame)

    # perform cross-validation
    print("-> Performing cross validation ...")
    image_dataset = MyDataset(data_frame)
    print("- dataset: {} images".format(len(image_dataset)))
    image_datasets = {}
    i, best_acc_folds, best_kapp_folds = 0, 0, 0
    init_time = time.time()

    skf = StratifiedKFold(n_splits=n_folds)
    # for train_idx, eval_idx in k_folds(n_splits=n_folds, subjects=len(image_dataset), frames=1):
    for train_idx, eval_idx in skf.split(image_dataset.image_paths, image_dataset.targets):
        print("\n-> Processing Fold {} ...".format(i+1))
        # create the train/eval datasets using the folds
        indices = {train_set_name: train_idx, eval_set_name: eval_idx}
        data_frames = {x: data_frame.iloc[indices[x]] for x in img_subset_names}
        image_datasets = {x: MyDataset(data_frames[x], data_transforms[x]) for x in img_subset_names}

        # load the train and eval sets in batches
        dataloaders = {x: torch.utils.data.DataLoader(dataset=image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in img_subset_names}
        dataset_sizes = {x: len(image_datasets[x]) for x in img_subset_names}
        class_names = image_datasets[train_set_name].labels
        n_classes = len(class_names)
        for x in img_subset_names:
            print("- {} set: {} images".format(x, dataset_sizes[x]))

        ###
        l = [os.path.splitext(os.path.basename(x))[0] for x in image_datasets[train_set_name].image_paths]
        # print(*l, sep=", ")
        # print(image_datasets[train_set_name].targets)
        # print(image_datasets[train_set_name].targets)
        print(collections.Counter(image_datasets[train_set_name].targets))
        ###

        # load the model
        pre_trained = "True" if train_mode in ['transfer-learning','fixed-feats'] else "False"
        model = load_model(model_name, pre_trained)
        
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
        optimizer = define_optimizer(optimizer_name, model, train_mode, learn_rate=learn_rate)

        # decay learn_rate by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # train the model
        model, train_results = train_model(model, [train_set_name,eval_set_name], loss_function, optimizer, scheduler, n_epochs)

        # save results for the current fold
        results_json = dict()
        results_json['input_dataset'] = input_dataset
        results_json['img_subset_names'] = img_subset_names
        results_json['n_samples_per_set'] = dict()
        for subset_name in img_subset_names:
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

        results_filename = os.path.join(output_dirname, "results_{}_fold_{}.json".format(output_suffix, i+1))
        fp = open(results_filename, 'w')
        json.dump(results_json, fp, sort_keys=False, indent=4)

        # save the best model
        if train_results['best_eval_acc'] > best_acc_folds:
            best_acc_folds = train_results['best_eval_acc']
            best_kappa_folds = train_results['best_eval_kappa']
            best_model_folds = copy.deepcopy(model.state_dict())

        i += 1

    # print best results
    time_elapsed = time.time() - init_time
    print()
    print('Cross-validation completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best from eval in all folds -> Accuracy: {:4f}, Kappa: {:4f}'.format(best_acc_folds, best_kappa_folds))
    
    # save the best PyTorch model
    model_filename = os.path.join(output_dirname, "model_{}.model".format(output_suffix))
    model.load_state_dict(best_model_folds)
    torch.save(model.state_dict(), model_filename)
    print("Best model saved in '{}".format(model_filename))