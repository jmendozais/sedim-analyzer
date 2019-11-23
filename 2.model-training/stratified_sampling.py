#!/usr/bin/python

# author: Cesar Castelo
# date: Nov 22, 2019
# description: Performs stratified sampling from a set of images and a set of image labels

import os
import argparse
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import collections

#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Performs stratified sampling from a set of images with their respective labels. It creates CSV files containing the train and test splits')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_csv', type=str, help='Input CSV containing the image filenames and labels', required=True)
    required_named.add_argument('-p', '--imgs_path', type=str, help='Path where the images are localized', required=True)
    required_named.add_argument('-o', '--output_dir', type=str, help='Output directory where the splits will be saved', required=True)
    parser.add_argument("--train_perc", type=float, default=0.5, help="Training percentage")
    parser.add_argument("--save_test_set", action="store_true", default=False, help="Whether or not to save the the test set")
    parser.add_argument("--save_imgs", action="store_true", default=False, help="Save the images in the output path")
    parser.add_argument("--save_imgs_pytorch_format", action="store_true", default=False, help="Save the images in the output path using the PyTorch name convention")
    args = parser.parse_args()

    # read input parameters
    input_csv = args.input_csv
    imgs_path = args.imgs_path
    output_dir = args.output_dir
    train_perc = args.train_perc
    save_test_set = args.save_test_set
    save_imgs = args.save_imgs
    save_imgs_pytorch_format = args.save_imgs_pytorch_format

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # read the input data
    data = pd.read_csv(input_csv, delimiter=',', header=0)
    X = list(data.iloc[:,0])
    y = list(data.iloc[:,1])

    # perform stratified sampling
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1.0-train_perc, random_state=0, stratify=y)
    
    # save the CSVs
    df = pd.DataFrame()
    df["Patch"] = [os.path.join(imgs_path, img) for img in x_train]
    df["Etiqueta"] = y_train
    df.to_csv(os.path.join(output_dir, "train.csv"), sep=",", index=False, header=True)
    print("- Train split created: {} images".format(len(x_train)))
    if save_test_set:
        df = pd.DataFrame()
        df["Patch"] = [os.path.join(imgs_path, img) for img in x_test]
        df["Etiqueta"] = y_test
        df.to_csv(os.path.join(output_dir, "test.csv"), sep=",", index=False, header=True)
        print("- Test split created: {} images".format(len(x_test)))
    
    # save the images
    if save_imgs:
        print("-> Saving the images ... ")
        # create subfolders
        train_dir = os.path.join(output_dir, "train")
        if not os.path.isdir(train_dir):
            os.makedirs(train_dir)
        if save_test_set:
            test_dir = os.path.join(output_dir, "test")
            if not os.path.isdir(test_dir):
                os.makedirs(test_dir)

        # save images in train/test folders using PyTorch format
        if save_imgs_pytorch_format:
            for c in set(y):
                if not os.path.isdir(os.path.join(train_dir, c)):
                    os.makedirs(os.path.join(train_dir, c))
            for i in range(len(x_train)):
                shutil.copyfile(os.path.join(imgs_path, x_train[i]), os.path.join(train_dir, y_train[i], x_train[i]))
            
            if save_test_set:
                for c in set(y):
                    if not os.path.isdir(os.path.join(test_dir, c)):
                        os.makedirs(os.path.join(test_dir, c))
                for i in range(len(x_test)):
                    shutil.copyfile(os.path.join(imgs_path, x_test[i]), os.path.join(test_dir, y_test[i], x_test[i]))
        # save images in train/test folders
        else:
            for img in x_train:
                shutil.copyfile(os.path.join(imgs_path, img), os.path.join(train_dir, img))
            if save_test_set:
                for img in x_test:
                    shutil.copyfile(os.path.join(imgs_path, img), os.path.join(test_dir, img))
