import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.impute import KNNImputer
from radiomics import featureextractor
import SimpleITK as sitk
from tqdm import tqdm 


df = pd.read_csv("D:/Virtual studio/MAI3004 test/hecktor2022_clinical_info_training.csv")
columns_to_drop = ["Tobacco", "Alcohol"]
df = df.drop(columns_to_drop, axis = 1)

path_img = r"D:\Virtual studio\MAI3004 test\data\imagesTr"
path_label = r"D:\Virtual studio\MAI3004 test\data\labelsTr"
path_labels_all = r"D:\Virtual studio\MAI3004 test\data\labels_all"
path_labels_tum_all = r"D:\Virtual studio\MAI3004 test\data\labels_tum_all"
path_labels_lymph_all = r"D:\Virtual studio\MAI3004 test\data\labels_lymph_all"
img_files = os.listdir(path_img)

extractor = featureextractor.RadiomicsFeatureExtractor()
#extractor.disableAllFeatures()
#feature_classes = ["firstorder", "shape", "glcm"]
#extractor.enableFeatureClassByName("firstorder")
ct_files = [x for x in os.listdir(path_img) if x.split("_")[-1]=="0000.nii.gz"]

items_to_pop = ['diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy', 'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet', 'diagnostics_Versions_Python', 'diagnostics_Configuration_Settings', 'diagnostics_Configuration_EnabledImageTypes', 'diagnostics_Image-original_Hash', 'diagnostics_Image-original_Dimensionality', 'diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size', 'diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum', 'diagnostics_Mask-original_Hash', 'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum', 'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass']
results_list = []
for img in tqdm(ct_files):
    ct_label_name = img.replace("_0000.nii.gz", ".nii.gz")
    image_path = path_img + "\\" + img
    label_path = path_labels_all + "\\"+ ct_label_name
    
    lab_name = os.path.join(path_label, ct_label_name)
    lab_img_sitk = sitk.ReadImage(label_path)
    lab_img_arr = sitk.GetArrayFromImage(lab_img_sitk)
    if lab_img_arr.max() > 0:
        result = extractor.execute(image_path, label_path)

        for i in items_to_pop:
            result.pop(i)
        patientid =  img.split("_")[0]
        result.update({'PatientID': patientid})
        results_list.append(result) 
    
df2 = pd.DataFrame.from_dict(results_list)
print(df2.head())

#df.join(df2, on=['PatientID'], how='left')
#df3 = pd.concat([df,df2], axis=1)
df = df.merge(
    df2,
    on='PatientID'
)
print(df.head())

df.to_csv('all_MRI.csv')

# Load clinical data
df = pd.read_csv("D:/Virtual studio/MAI3004 test/hecktor2022_clinical_info_training.csv")
columns_to_drop = ["Tobacco", "Alcohol"]
df = df.drop(columns_to_drop, axis=1)

# Define paths for PET scans and labels
path_img = r"D:\Virtual studio\MAI3004 test\data\imagesTr"
path_label = r"D:\Virtual studio\MAI3004 test\data\labelsTr"
path_labels_all = r"D:\Virtual studio\MAI3004 test\data\labels_all"

# Get PET files (adjust the filename filter for PET scans)
pet_files = [x for x in os.listdir(path_img) if x.split("_")[-1] == "0001.nii.gz"]

# Initialize the radiomics feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

# List of diagnostic metadata to exclude
items_to_pop = [
    'diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy',
    'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet',
    'diagnostics_Versions_Python', 'diagnostics_Configuration_Settings',
    'diagnostics_Configuration_EnabledImageTypes', 'diagnostics_Image-original_Hash',
    'diagnostics_Image-original_Dimensionality', 'diagnostics_Image-original_Spacing',
    'diagnostics_Image-original_Size', 'diagnostics_Image-original_Mean',
    'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum',
    'diagnostics_Mask-original_Hash', 'diagnostics_Mask-original_Spacing',
    'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox',
    'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum',
    'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass'
]

# Extract radiomics features from PET scans
results_list = []
for img in tqdm(pet_files):
    pet_label_name = img.replace("_0001.nii.gz", ".nii.gz")  # Update label naming for PET scans
    image_path = os.path.join(path_img, img)
    label_path = os.path.join(path_labels_all, pet_label_name)

    lab_img_sitk = sitk.ReadImage(label_path)
    lab_img_arr = sitk.GetArrayFromImage(lab_img_sitk)
    
    if lab_img_arr.max() > 0:  # Ensure label contains relevant data
        result = extractor.execute(image_path, label_path)
        
        # Remove diagnostic metadata
        for i in items_to_pop:
            result.pop(i, None)
        
        # Add Patient ID to the results
        patientid = img.split("_")[0]
        result.update({'PatientID': patientid})
        results_list.append(result)

# Convert results to DataFrame
df2 = pd.DataFrame.from_dict(results_list)
print(df2.head())

# Merge clinical and radiomic data
df = df.merge(
    df2,
    on='PatientID'
)
print(df.head())

# Save the final DataFrame to a CSV file
df.to_csv('All_PET.csv', index=False)
