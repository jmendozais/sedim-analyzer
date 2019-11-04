import argparse
import numpy as np
import subprocess
import os, sys
import cv2

from PIL import Image
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk, opening
from skimage.filters import threshold_otsu, threshold_minimum
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.transform import resize
import time

class SampleTemplate:
    def __init__(self):
        self.roi = None
        self.shape = None
        
class ROILocalizer:
    def __init__(self):
        self.templates = []
        
    def verify_membership(self, img, template):
        # TODO: Implement verification by histograms. 
	#	There are images with the same shape but different ROI localization.
        return True
    
    def __localize(self, image, noise_segments_percent=0.05, bridge_width=9):
        '''
        Return: Bounding box of the ROI in the format (tl_x, tl_y, br_x, br_y). tl stands for the top left corner,
        br stands for the botton right corner
        '''
        factor = 1
        if image.shape[0] > 1200:
            factor = image.shape[0]/1200.0
            new_size = (int(image.shape[0]/factor), int(image.shape[1]/factor))
            image = resize(image, new_size, anti_aliasing=True)
            
        gray_image = rgb2gray(image)
        gray_image = np.array(gray_image * 256).astype(np.uint8)

        # Perform binary segmentation based on entropy
        entropy_image = entropy(gray_image, disk(5))
        thresh = threshold_otsu(entropy_image)
        binary = entropy_image > thresh
        
        # Remove bridges
        binary = opening(binary, disk(bridge_width//2))

        # Remove small objects
        label_objects, nb_labels = ndi.label(binary)
        sizes = np.bincount(label_objects.ravel())
        h, w, _ = image.shape
        mask_sizes = sizes > h*w*noise_segments_percent
        mask_sizes[0] = 0
        segments_cleaned = mask_sizes[label_objects].astype(np.int)

        # Find bounding box of the ROI and scaling to the original size
        props = regionprops(segments_cleaned)
        roi = props[0].bbox
        roi = (int(roi[0]*factor), int(roi[1]*factor), int(roi[2]*factor), int(roi[3]*factor))
        
        return roi 

    def localize(self, img):
        roi = None
        for template in self.templates:
            if img.shape == template.shape:
                if self.verify_membership(img, template):
                    roi = template.roi
                    shape = template.shape
        
        if roi == None:
            roi = self.__localize(img)
            
            template = SampleTemplate()
            template.roi = roi
            template.shape = img.shape
            self.templates.append(template)
        return roi

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

def process_img(img, filename, output_folder, pit_id, localizer):
    roi = localizer.localize(img)
    img = img[roi[0]:roi[2],roi[1]:roi[3]]
    split_image(img, filename, output_folder)
    return


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
    else:
        roi = localizer.localize(img)
        img = img[roi[0]:roi[2],roi[1]:roi[3]]
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
    localizer = ROILocalizer()
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
        process_img(img, file, output_folder, pit_id, localizer)
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
