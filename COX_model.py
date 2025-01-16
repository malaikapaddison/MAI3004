import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

df = pd.read_csv("D:/Virtual studio/MAI3004 test/all_PET.csv")

#X = df.drop(["RFS"], axis=1)
#y = df['RFS']
#split data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#best_features = SelectKBest(score_func = f_classif,k = 'all')
#fit = best_features.fit(X_train,y_train)

#featureScores = pd.DataFrame(data = fit.scores_,index = list(X.columns),columns = ['ANOVA Score'])
#top_features = featureScores.sort_values(by='ANOVA Score', ascending=False).head(10).index

#Survival_time = df["RFS"]
#top_features.insert(1, "RFS", Survival_time)
#top_features = top
#Relapsed = df["Relapse"]
#top_features.insert(2, "Relapse", Relapsed)
#top_features = top_features.insert(1, "RFS")
#top_features = top_features.insert(2, "Relapse")
features_to_drop = ['original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_gldm_SmallDependenceEmphasis', 'original_gldm_SmallDependenceHighGrayLevelEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis', 'original_glszm_SizeZoneNonUniformity', 'original_glszm_ZonePercentage', 'original_ngtdm_Contrast']
df.drop(features_to_drop, axis=1)
cph = CoxPHFitter(penalizer=0.2)
#cph.fit(df, duration_col='RFS', event_col='Relapse')

#print(cph.concordance_index_)
#print(top_features)
print(np.mean(k_fold_cross_validation(cph, df, duration_col='RFS', event_col='Relapse', scoring_method="concordance_index" , seed=1)))