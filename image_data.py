import nibabel as nib
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm 


path_img = r"D:\Virtual studio\MAI3004 test\data\imagesTr"
path_label = r"D:\Virtual studio\MAI3004 test\data\labelsTr"
store_label_tum_all = r"D:\Virtual studio\MAI3004 test\data\labels_tum_all"

img_files = os.listdir(path_label)

for img in tqdm(img_files):
    lab_name = os.path.join(path_label, img)
    lab_img_sitk = sitk.ReadImage(lab_name)

    lab_img_arr = sitk.GetArrayFromImage(lab_img_sitk)
    lab_img_arr_tum = (lab_img_arr == 1) * 1
    lab_img_arr_tum = lab_img_arr_tum.astype("uint8")
    lab_mod_img = sitk.GetImageFromArray(lab_img_arr_tum)
    lab_mod_img.CopyInformation(lab_img_sitk)

    sitk.WriteImage(lab_mod_img, os.path.join(store_label_tum_all, img))



"""
for img in tqdm(img_files):
    ct_img_name = os.path.join(path_img, img.replace(".nii.gz", "_0000.nii.gz"))
    pet_img_name = os.path.join(path_img, img.replace(".nii.gz", "_0001.nii.gz"))
    lab_name = os.path.join(path_img, img)
"""




