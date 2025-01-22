import nibabel as nib
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm 


path_img = r"D:\Virtual studio\MAI3004 test\data\imagesTr"
path_label = r"D:\Virtual studio\MAI3004 test\data\labelsTr"
store_label_tum_all = r"D:\Virtual studio\MAI3004 test\data\labels_all"

ct_files = [x for x in os.listdir(path_img) if x.split("_")[-1]=="0000.nii.gz"]
print(ct_files)

pet_files= [x for x in os.listdir(path_img) if x.split("_")[-1]=="0001.nii.gz"]
print(pet_files)
