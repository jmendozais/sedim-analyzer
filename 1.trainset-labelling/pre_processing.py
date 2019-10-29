import argparse
import numpy as np
import subprocess
import os, sys
import cv2

def load_image(filename) :
    return cv2.imread(filename)

def save_image(img, filename) :
    cv2.imwrite(filename, img)

def split_image(img, filename, output_folder):
    basename = os.path.basename(filename)
    filename_no_ext = os.path.splitext(basename)[0]

    heigth = img.shape[0]//4
    
    img_1 = img[:heigth,:,:]
    save_image(img_1, os.path.join(output_folder, filename_no_ext + "_patch_1.png"))

    img_2 = img[heigth:heigth*2,:,:]
    save_image(img_2, os.path.join(output_folder, filename_no_ext + "_patch_2.png"))

    img_3 = img[heigth*2:heigth*3,:,:]
    save_image(img_3, os.path.join(output_folder, filename_no_ext + "_patch_3.png"))

    img_4 = img[heigth*3:,:,:]
    save_image(img_4, os.path.join(output_folder, filename_no_ext + "_patch_4.png"))

def process_img(img, filename, output_folder, pit_id):
    if pit_id == 1:
        img = img[:, 50:-50, :]
        split_image(img, filename, output_folder)
    elif pit_id == 2:
        img = img[:, 524:1136, :]
        split_image(img, filename, output_folder)
    elif pit_id == 3:
        img = img[:, 524:1136, :]
        split_image(img, filename, output_folder)
    elif pit_id == 4:
        img = img[352:1938, 122:667, :]
        split_image(img, filename, output_folder)
    elif pit_id == 5:
        img = img[25:2090, 472:1066, :]
        split_image(img, filename, output_folder)
    elif pit_id == 6:
        img = img[31:509, 119:262, :]
        split_image(img, filename, output_folder)
    elif pit_id == 7:
        img = img[70:2060, 458:1050, :]
        split_image(img, filename, output_folder)
    elif pit_id == 8:
        img = img[352:1938, 132:637, :]
        split_image(img, filename, output_folder)
    elif pit_id == 9:
        img = img[45:2075, 551:1135, :]
        split_image(img, filename, output_folder)
    elif pit_id == 10:
        img = img[80:2062, 472:1076, :]
        split_image(img, filename, output_folder)

#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Applies the pre-processing steps to the images')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_folder', type=str, help='Folder containing the images', required=True)
    required_named.add_argument('-o', '--output_folder', type=str, help='Output folder', required=True)
#    parser.add_argument("--num_pits", type=int, default=10, help="Number of pits")
    args = parser.parse_args()

    # read input parameters
    input_folder = args.input_folder
    output_folder = args.output_folder

    # read the files in the directory
    files = os.listdir(input_folder)
    files.sort()
    files_zip = zip(range(1, len(files)+1), files)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # process each image
    n_imgs_per_pit = {}
    for i, file in files_zip:
        # split the filename
        basename = os.path.basename(file)
        filename_no_ext = os.path.splitext(basename)[0]
        token = filename_no_ext.split('_')
        pit_id = int(token[1])

        # count the number of images per pit
        if not pit_id in n_imgs_per_pit:
            n_imgs_per_pit[pit_id] = 0
        n_imgs_per_pit[pit_id] += 1

        # read image
        img = load_image(os.path.join(input_folder, file))
        
        # apply a different procedure according to the number of pit
        print("Processing image {}/{} ...\r".format(i, len(files)), end="")
        process_img(img, file, output_folder, pit_id)
    # print("Processing image {}/{} ...".format(len(files)+1, len(files)+1))
    print()

    # print results summary
    n_patches = 0
    print("Images processed by pit:")
    for i in range(1, max(n_imgs_per_pit)+1):
        if i in n_imgs_per_pit:
            print("Pit {}: {} images (created {} patches)".format(i, n_imgs_per_pit[i], n_imgs_per_pit[i]*4))
            n_patches += n_imgs_per_pit[i]*4
    print("Patches created: {}".format(n_patches))