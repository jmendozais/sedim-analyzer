# Author: Cesar Castelo, Julio Mendoza
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
        self.descriptor = None
        
class ROILocalizer:
    def __init__(self):
        self.templates = []
        self.dists = [] 
    
    def compute_descriptor(self, image, roi):
        mean_cols = np.mean(image, axis=(0, 2))
        mean_rows = np.mean(image, axis=(1, 2))
        descriptor = np.concatenate((mean_cols[:roi[1]], mean_cols[roi[3]:], mean_rows[:roi[0]], mean_rows[roi[2]:]))
        return descriptor 
    
    def verify_membership(self, image, template):
        if image.shape == template.shape:
            roi_w =  template.roi[3] - template.roi[1]
            descriptor = self.compute_descriptor(image, template.roi)
            dist_mean = np.mean(np.abs(descriptor - template.descriptor)) 
            self.dists.append(dist_mean)
            return dist_mean < 3
        return False
    
    def __localize(self, image, noise_segments_percent=0.05, bridge_width=15):
        '''
        Return: Bounding box of the ROI in the format (tl_x, tl_y, br_x, br_y). tl stands for the top left corner,
        br stands for the botton right corner
        '''
        factor = 1
        target_shape = 800
        if image.shape[0] > target_shape:
            factor = image.shape[0]/target_shape
            new_size = (int(image.shape[0]/factor), int(image.shape[1]/factor))
            image = resize(image, new_size, anti_aliasing=True)
            
        gray_image = rgb2gray(image)
        gray_image = np.array(gray_image * 256).astype(np.uint8)

        # Perform binary segmentation based on entropy if needed
        entropy_image = entropy(gray_image, disk(4))
        if np.std(entropy_image) > 1.0:
            thresh = threshold_otsu(entropy_image)
        else:
            thresh = 0.0
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
            if self.verify_membership(img, template):
                roi = template.roi
                shape = template.shape
                break
        
        if roi == None:
            roi = self.__localize(img)

            template = SampleTemplate()
            template.roi = roi
            template.shape = img.shape
            template.descriptor = self.compute_descriptor(img, roi)
            self.templates.append(template)
        return roi
     
def load_image(filename) :
    return cv2.imread(filename)

def save_image(img, filename) :
    cv2.imwrite(filename, img)

def save_results(results, output_folder):
    for result in results:
        basename, img = result
        basename = basename[:-3] + "png"
        save_image(img, os.path.join(output_folder, basename))

def process_img(img, pit_id, localizer):
    if pit_id == 1:
        img = img[:, 50:-50, :]
    elif pit_id == 2:
        img = img[:, 524:1136, :]
    elif pit_id == 3:
        img = img[:, 524:1136, :]
    elif pit_id == 4:
        img = img[352:1938, 122:667, :]
    elif pit_id == 5:
        img = img[25:2090, 472:1066, :]
    elif pit_id == 6:
        img = img[31:509, 119:262, :]
    elif pit_id == 7:
        img = img[70:2060, 458:1050, :]
    elif pit_id == 8:
        img = img[352:1938, 132:637, :]
    elif pit_id == 9:
        img = img[45:2075, 551:1135, :]
    elif pit_id == 10:
        img = img[80:2062, 472:1076, :]
    else:
        roi = localizer.localize(img)
        img = img[roi[0]:roi[2],roi[1]:roi[3]]

    return img

def get_well_id_from_filename(filename):
    basename = os.path.basename(filename)
    filename_no_ext = os.path.splitext(basename)[0]

    token = filename_no_ext.split('_')
    if token != None:
        pit_id = int(token[1])
    else:
        pit_id = -1
    return pit_id


#===============================================#
# Function to be called from outside the script
#===============================================#
def preprocess_data(input_folder, output_folder):
    print("Iniciando preprocesamiento")
    # T1 : Loop para iniciar el preprocesamiento
    start = time.time()

    # read the files in the directory
    files = os.listdir(input_folder)
    files.sort()
    files_zip = zip(range(1, len(files)+1), files)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # process each image
    n_imgs_per_pit = {}
    localizer = ROILocalizer()
    results_list = []

    for i, file in files_zip:
        # read image
        img = load_image(os.path.join(input_folder, file))

        # split the filename
        pit_id = get_well_id_from_filename(file)

        # count the number of images per pit
        if not pit_id in n_imgs_per_pit:
            n_imgs_per_pit[pit_id] = 0
        n_imgs_per_pit[pit_id] += 1
        
        # apply a different procedure according to the number of pit
        print("Procesando imagen {}/{} ...\r".format(i, len(files)), end="")
        processed_img = process_img(img, pit_id, localizer)
        results_list.append((basename, processed_img))

    # T2 : Preprocesamient finalizado
    total_time = time.time() - start
    print("Preprocesamiento finalizado: ", total_time)
    print("Guardando resultados ...")

    # Guardar los resultados
    save_results(results_list, output_folder)

    total = 0
    print("Reporte: imagenes procesadas por pozo:")
    for pozo_id in n_imgs_per_pit:
        print("Pozo {}: {} imagenes".format(pozo_id, n_imgs_per_pit[pozo_id]))
        total += n_imgs_per_pit[pozo_id]
    print("Imagenes creadas: {}".format(total))
    assert total == len(files)

#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Applies the pre-processing steps to the images')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_folder', type=str, help='Folder containing the images', required=True)
    required_named.add_argument('-o', '--output_folder', type=str, help='Output folder', required=True)
#   parser.add_argument("--num_pits", type=int, default=10, help="Number of pits")
    args = parser.parse_args()

    # read input parameters
    input_folder = args.input_folder
    output_folder = args.output_folder

    # execute the pre-processing procedure
    preprocess_data(input_folder, output_folder)
    