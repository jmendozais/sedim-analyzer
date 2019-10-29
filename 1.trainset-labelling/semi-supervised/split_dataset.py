import argparse
import numpy as np
import subprocess
import os, sys
from shutil import copyfile
import random

#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Splits a supervised image set into training and validation sets')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_folder', type=str, help='Folder containing the images (organized in class folders)', required=True)
    required_named.add_argument('-o', '--output_folder', type=str, help='Output folder', required=True)
    required_named.add_argument('-p', '--perc_train', type=float, help='Percentage for the training set', required=True)
    args = parser.parse_args()

    # read input parameters
    input_folder = args.input_folder
    output_folder = args.output_folder
    perc_train = args.perc_train

    # create the output folders
    train_output_folder = os.path.join(output_folder, "train")
    val_output_folder = os.path.join(output_folder, "val")
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    if not os.path.isdir(train_output_folder):
        os.makedirs(train_output_folder)
    if not os.path.isdir(val_output_folder):
        os.makedirs(val_output_folder)

    # read the input folder and process the class folders
    class_dirs = os.listdir(input_folder)
    class_dirs.sort()
    for class_dir in class_dirs:
        print("Processing class '{}'".format(class_dir))
        # copy the files from the folder
        files = os.listdir(os.path.join(input_folder, class_dir))
        random.shuffle(files)
        # train set
        if not os.path.isdir(os.path.join(train_output_folder, class_dir)):
            os.makedirs(os.path.join(train_output_folder, class_dir))
        for i in range(int(perc_train*len(files))):
            print("- train set: image {}/{} ...\r".format(i+1, int(perc_train*len(files))), end="")
            src = os.path.join(input_folder, class_dir, files[i])
            dst = os.path.join(train_output_folder, class_dir, files[i])
            copyfile(src, dst)
        print()
        # val set
        if not os.path.isdir(os.path.join(val_output_folder, class_dir)):
            os.makedirs(os.path.join(val_output_folder, class_dir))
        for i in range(int(perc_train*len(files)), len(files)):
            print("- val set: image {}/{} ...\r".format(i-int(perc_train*len(files))+1, len(files)-int(perc_train*len(files))), end="")
            src = os.path.join(input_folder, class_dir, files[i])
            dst = os.path.join(val_output_folder, class_dir, files[i])
            copyfile(src, dst)
        print()
