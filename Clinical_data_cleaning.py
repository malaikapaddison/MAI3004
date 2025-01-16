import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.impute import KNNImputer

df = pd.read_csv("D:/Virtual studio/MAI3004 test/all_PET.csv")
columns_to_drop = ["PatientID"]
df = df.drop(columns_to_drop, axis = 1)

print(df.columns)

print(df.head())

df[["Surgery","Performance status", "HPV status (0=-, 1=+)"]] = df[["Surgery", "Performance status","HPV status (0=-, 1=+)"]].replace("without", np.nan)
df = df.replace({'Gender' : { 'M' : 0, 'F' : 1}})
imputer = KNNImputer(n_neighbors=5).set_output(transform = "pandas")
imputed_data = imputer.fit_transform(df)
df[["Performance status", "HPV status (0=-, 1=+)"]] = imputed_data[["Performance status", "HPV status (0=-, 1=+)"]].round().astype(int)


imputed_data_surgery = imputer.fit_transform(df[["Surgery","Performance status", "Chemotherapy", "Age", "Weight"]])
df["Surgery"] = imputed_data_surgery["Surgery"].round().astype(int)

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

print(df.head())

df.to_csv('All_PET.csv')