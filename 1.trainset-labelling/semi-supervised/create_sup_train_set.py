import argparse
import numpy as np
import subprocess
import os, sys
import cv2
import csv
import pandas as pd
from shutil import copyfile

def load_image(filename) :
    return cv2.imread(filename)

def save_image(img, filename) :
    cv2.imwrite(filename, img)

def copy_patch(data, img_id, patch_id, input_folder, output_folder):
    filename = "{}_patch_{}.png".format(data.iloc[img_id,0], patch_id)
    src = os.path.join(input_folder, filename)
    dst = os.path.join(output_folder, data.iloc[img_id,patch_id], filename)
    copyfile(src, dst)


#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Creates the supervised training set')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_folder', type=str, help='Folder containing the pre-processed images (patches)', required=True)
    required_named.add_argument('-l', '--input_labels', type=str, help='CSV containing the labels for each image', required=True)
    required_named.add_argument('-o', '--output_folder', type=str, help='Output folder', required=True)
    args = parser.parse_args()

    # read input parameters
    input_folder = args.input_folder
    input_labels = args.input_labels
    output_folder = args.output_folder

    # create the output folders
    sup_output_folder = os.path.join(output_folder, "supervised")
    unsup_output_folder = os.path.join(output_folder, "unsupervised")
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    if not os.path.isdir(sup_output_folder):
        os.makedirs(sup_output_folder)
    if not os.path.isdir(unsup_output_folder):
        os.makedirs(unsup_output_folder)

    # read the csv containing the labels
    data = pd.read_csv(input_labels, sep=";")

    # create the folders for each class
    classes = set(list(data.iloc[:,1]) + list(data.iloc[:,2]) + list(data.iloc[:,3]) + list(data.iloc[:,4]))
    for c in classes:
        if not os.path.isdir(os.path.join(sup_output_folder, c)):
            os.makedirs(os.path.join(sup_output_folder, c))

    # copy the labelled images to the folders
    print("--> Copying the labelled images ...")
    for i in range(len(data)):
        print("Processing image {}/{} ...\r".format(i+1, len(data)), end="")
        copy_patch(data, i, 1, input_folder, sup_output_folder)
        copy_patch(data, i, 2, input_folder, sup_output_folder)
        copy_patch(data, i, 3, input_folder, sup_output_folder)
        copy_patch(data, i, 4, input_folder, sup_output_folder)
    print()

    # copy the unlabelled images to the folders
    print("--> Copying the unlabelled images ...")
    files = os.listdir(input_folder)
    files.sort()

    n_files = len(files) - len(list(data.iloc[:,0]))*4
    i = 1
    for file in files:
        filename = os.path.splitext(os.path.basename(file))[0].split("_")
        filename = "{}_{}_{}_{}".format(filename[0],filename[1],filename[2],filename[3])
        if not filename in list(data.iloc[:,0]):
            print("Processing image {}/{} ...\r".format(i, n_files), end="")
            src = os.path.join(input_folder, file)
            dst = os.path.join(unsup_output_folder, file)
            copyfile(src, dst)
            i += 1
    print()

