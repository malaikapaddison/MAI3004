import pandas as pd
import numpy as np
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
import os
from radiomics import featureextractor

df = pd.read_csv("D:/Virtual studio/MAI3004 test/clinical_info.csv")

print(df.columns)

print(df.head())

print(df.Tobacco.value_counts())
df[['Tobacco', "Alcohol", "Surgery","Performance status", "HPV status (0=-, 1=+)"]] = df[['Tobacco', "Alcohol", "Surgery", "Performance status","HPV status (0=-, 1=+)"]].replace("without", 0)
df[['Tobacco', "Alcohol", "Surgery", "Performance status",  "HPV status (0=-, 1=+)"]] = df[['Tobacco', "Alcohol", "Surgery", "Performance status", "HPV status (0=-, 1=+)"]].astype(int)
#here 'without' was changed to 0, however we should treat them as missing values
print(df.Tobacco.value_counts())
print(df["HPV status (0=-, 1=+)"].value_counts())

print(df.Weight.max())

def identify_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outlier_indices = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)].index.tolist()
    outlier_values = df.loc[outlier_indices, column_name].tolist()
    return outlier_indices, outlier_values

outlier_indices, outlier_values = identify_outliers(df,"Age")
print(outlier_values)
print(outlier_indices)

df.loc[outlier_indices, "Age"] = np.nan
mean_age = df.Age.mean()
df.loc[outlier_indices, "Age"] = mean_age

outlier_indices, outlier_values = identify_outliers(df,"Age")

print(outlier_values)
print(outlier_indices)

outlier_indices, outlier_values = identify_outliers(df,"Weight")
print(outlier_values)
print(outlier_indices)

#Only the outlier with a weight of 34 seems to be impossible and was removed
weight_index = [172]
df.loc[weight_index, "Weight"] = np.nan
mean_weight = df.Weight.mean()
df.loc[weight_index, "Weight"] = mean_weight

outlier_indices, outlier_values = identify_outliers(df,"Weight")
print(outlier_values)

print(df.Gender.value_counts())

#clinical feature vizualisations

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
sns.histplot(ax=axes[0,0],data=df, x="Gender", hue="Gender")
sns.histplot(ax=axes[0,1],data=df, x="Surgery")
sns.histplot(ax=axes[0,2],data=df, x="Alcohol")
sns.histplot(ax=axes[0,3],data=df, x="HPV status (0=-, 1=+)")
sns.histplot(ax=axes[1,0],data=df, x="Chemotherapy")
sns.histplot(ax=axes[1,1],data=df, x="Weight")
sns.histplot(ax=axes[1,2],data=df, x="Age")
sns.histplot(ax=axes[1,3],data=df, x="Performance status")
plt.show()
#sns.boxplot

#sns.histplot(data=df, x="Weight")
#plt.show()

print(df.isnull().sum())
print(len(df.PatientID.unique()))
print(df.shape)


print(os.getcwd())
#testCase = 'HandNCancer'
#dataDir = os.path.join(os.getcwd(), "..",  "data")

#imagePath = os.path.join(dataDir, testCase + "imageTR.zip")
#labelpath = os.path.join(dataDir, testCase + "labelsTR.zip")

path_img = r"D:\Virtual studio\MAI3004 test\data\imagesTr"
path_label = r"D:\Virtual studio\MAI3004 test\data\labelsTr"
store_label_all = r"D:\Virtual studio\MAI3004 test\data\labels_all"
path_labels_tum_all = r"D:\Virtual studio\MAI3004 test\data\labels_all_tum_all"
img_files = os.listdir(path_img)

extractor  = featureextractor.RadiomicsFeaturesExtractor()

ct_files = [x for x in os.listdir(path_img) if x.split("_")[-1]=="0000.nii.gz"]
print(ct_files)

for img in tqdm(ct_files):
    ct_img_name = os.path.join(path_img, img.replace(".nii.gz", "_0000.nii.gz"))
    result = extractor.execute(path_img + img, path_labels_tum_all + ct_img_name)

