import pandas as pd
import numpy as np
import zipfile

df = pd.read_csv("D:/Virtual studio/MAI3004 test/clinical_info.csv")

print(df.columns)

print(df.head())

print(df.Tobacco.value_counts())
df[['Tobacco', "Alcohol", "Surgery","Performance status", "HPV status (0=-, 1=+)"]] = df[['Tobacco', "Alcohol", "Surgery", "Performance status","HPV status (0=-, 1=+)"]].replace("without", 0)
df[['Tobacco', "Alcohol", "Surgery", "Performance status",  "HPV status (0=-, 1=+)"]] = df[['Tobacco', "Alcohol", "Surgery", "Performance status", "HPV status (0=-, 1=+)"]].astype(int)

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
#Only the outlier with a weight of 34 seems to be impossible
weight_index = [172]
df.loc[weight_index, "Weight"] = np.nan
mean_weight = df.Weight.mean()
df.loc[weight_index, "Weight"] = mean_weight

outlier_indices, outlier_values = identify_outliers(df,"Weight")
print(outlier_values)

print(df.Gender.value_counts())