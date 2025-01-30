import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.impute import KNNImputer
from radiomics import featureextractor
import SimpleITK as sitk
from tqdm import tqdm 


path_label = r"D:\Virtual studio\MAI3004 test\data\labelsTr"
path_label_tum_all = r"D:\Virtual studio\MAI3004 test\data\labels_tum_all"
store_label_Largest_tum = r"D:\Virtual studio\MAI3004 test\data\Largest_tum"


img_files = os.listdir(path_label_tum_all)

for img in tqdm(img_files):

    # Step 1: Load the image
    lab_name = os.path.join(path_label_tum_all, img)
    input_image = sitk.ReadImage(lab_name)  # Replace with your image path
    binary_image = sitk.BinaryThreshold(input_image, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)

    # Step 2: Connected Component Analysis
    connected_components = sitk.ConnectedComponent(binary_image)

    # Step 3: Relabel components by size
    relabeled_components = sitk.RelabelComponent(connected_components, sortByObjectSize=True)

    # Step 4: Extract the largest component (label 1 after relabeling)
    largest_component = sitk.BinaryThreshold(relabeled_components, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)

    # Step 5: Calculate the volume of the largest component
    largest_volume = sitk.GetArrayViewFromImage(largest_component).sum()
    print(f"Volume of the largest component: {largest_volume} voxels")

    # Save the largest component as an output image (optional)
    sitk.WriteImage(largest_component, os.path.join(store_label_Largest_tum, img))