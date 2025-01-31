import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.impute import KNNImputer

#load the dataframe
df = pd.read_csv("D:/Virtual studio/MAI3004 test/all_PET.csv")

#drop the PatientID column
columns_to_drop = ["PatientID"]
df = df.drop(columns_to_drop, axis = 1)

#replace the without values with a nan value for imputation
df[["Surgery","Performance status", "HPV status (0=-, 1=+)"]] = df[["Surgery", "Performance status","HPV status (0=-, 1=+)"]].replace("without", np.nan)

#encode the Gender column with 0 for male and 1 for female
df = df.replace({'Gender' : { 'M' : 0, 'F' : 1}})

#Impute the performance status and HPV status using K-nearest-neighbours where K is 5
imputer = KNNImputer(n_neighbors=5).set_output(transform = "pandas")
imputed_data = imputer.fit_transform(df)
df[["Performance status", "HPV status (0=-, 1=+)"]] = imputed_data[["Performance status", "HPV status (0=-, 1=+)"]].round().astype(int)

#Imput the surgery status using what we thought were appropriate data columns
imputed_data_surgery = imputer.fit_transform(df[["Surgery","Performance status", "Chemotherapy", "Age", "Weight"]])
df["Surgery"] = imputed_data_surgery["Surgery"].round().astype(int)

#Finds the outliers in a specific column in the dataframe using the interquartile range
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

#Identify the outliers for age
outlier_indices, outlier_values = identify_outliers(df,"Age")
print(outlier_values)
print(outlier_indices)

#replace the outliers with nan and then impute them with the mean age
df.loc[outlier_indices, "Age"] = np.nan
mean_age = df.Age.mean()
df.loc[outlier_indices, "Age"] = mean_age

# identify the outliers for weight
outlier_indices, outlier_values = identify_outliers(df,"Weight")
print(outlier_values)
print(outlier_indices)

#Only the outlier with a weight of 34 seems to be impossible and was removed
weight_index = [172]
df.loc[weight_index, "Weight"] = np.nan
mean_weight = df.Weight.mean()
df.loc[weight_index, "Weight"] = mean_weight

#save the cleaned df to csv
df.to_csv('All_PET.csv')
