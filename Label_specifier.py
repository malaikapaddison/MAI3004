import nibabel as nib
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm 

# selects the paths to the images and the where the new label images will be stored
path_img = r"D:\Virtual studio\MAI3004 test\data\imagesTr"
path_label = r"D:\Virtual studio\MAI3004 test\data\labelsTr"
store_label_lymph_all = r"D:\Virtual studio\MAI3004 test\data\labels_lymph_all"

#creates a list of all the labels
img_files = os.listdir(path_label)

#divides the labels based on whether the img label array is equal to either 1 for tumours or 2 for lymph nodes
for img in tqdm(img_files):
    lab_name = os.path.join(path_label, img)
    lab_img_sitk = sitk.ReadImage(lab_name)

    lab_img_arr = sitk.GetArrayFromImage(lab_img_sitk)
    lab_img_arr_tum = (lab_img_arr == 2) * 1
    lab_img_arr_tum = lab_img_arr_tum.astype("uint8")
    lab_mod_img = sitk.GetImageFromArray(lab_img_arr_tum)
    lab_mod_img.CopyInformation(lab_img_sitk)
    #saves the new label file in the specified location
    sitk.WriteImage(lab_mod_img, os.path.join(store_label_lymph_all, img))
