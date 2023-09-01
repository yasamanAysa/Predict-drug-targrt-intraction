import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
#==================================================================
#preprocessing data: dimention reduction feature of drugs and feature of targets:
df_drugs = pd.read_csv('dataWithHeader/featureDrugWithHeader.csv')
print(df_drugs)

df_targets = pd.read_csv('dataWithHeader/featureTargetWithHeader.csv')
print(df_targets)
#==================================================================================
# pca before merge:
# find d for 0.90 dimention reduction:
pca = PCA()
pca.fit(df_drugs)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d_drug = np.argmax(cumsum >= 0.90) + 1
print(d_drug)
# pca = PCA()
pca.fit(df_targets)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d_target = np.argmax(cumsum >= 0.90) + 1
print(d_target)
# d_drug  :  38
# d_target  :  160

n_components_drug = 38
n_components_target = 160

pca_drug = PCA(n_components_drug)
pca_target = PCA(n_components_target)

drugs_reduce = pca_drug.fit_transform(df_drugs)
targets_reduce = pca_target.fit_transform(df_targets)

print('Cumulative explained variation for {} principal components: {}'
      .format(n_components_drug, np.sum(pca_drug.explained_variance_ratio_)))
print('Cumulative explained variation for {} principal components: {}'
      .format(n_components_target, np.sum(pca_target.explained_variance_ratio_)))


# print(type(df_drugs))
#convert ndarray to df:
df_drugs_reduce = pd.DataFrame(data=drugs_reduce)
print(df_drugs_reduce)
df_targets_reduce = pd.DataFrame(data=targets_reduce)
print(df_targets_reduce)
#==========================================================
# write to csv:
df_drugs_reduce.to_csv (r'df_drugs_reduce.csv', index = False)
df_targets_reduce.to_csv (r'df_targets_reduce.csv', index = False)
#=================================================================================