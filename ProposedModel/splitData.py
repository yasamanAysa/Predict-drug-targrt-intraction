# 1:
#   1.1: split train/test
#   1.2: Separate majority and minority classes
#   1.3: pca
#   1.4: merge minority class(merge feature of drug and target):
#   1.5: go to 2.clustringMajority.py

#======================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
#===============================================================

df = pd.read_csv('files/df.csv')
#===============================================================
#split test/train:
X_train, X_test, y_train, y_test = train_test_split(df, df['interaction'], test_size = 0.15,
                                                    stratify=df['interaction'], random_state=50)
# print(X_train['interaction'].value_counts())
# print(X_train)
print(X_test['interaction'].value_counts())
print(X_test)
# print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
print(y_test)
#========================================================
# write test set to csv:
X_test.to_csv("X_test_index.csv",index = False)
y_test.to_csv("y_test.csv",index = False)
#========================================================
# Separate majority and minority classes
df_train_majority = X_train[X_train['interaction'] == 0]
df_train_minority = X_train[X_train['interaction'] == 1]
#=============================================================
# merge feature drug and target in minority:
df_drugs_reduce = pd.read_csv('reducedFeaturesDrugs.csv')
print(df_drugs_reduce)
df_targets_reduce = pd.read_csv('reducedFeaturesTargets.csvv')
print(df_targets_reduce)

#=========================================================
# merge minority class:
df_minority_train_index = df_train_minority[['drug_id','target_id']]
print(df_minority_train_index)
minority_train_index_numpy = df_minority_train_index.to_numpy()
print(minority_train_index_numpy)
print(len(minority_train_index_numpy))
def mergeDrugTarget(index_numpy,df_drugs_reduce,df_targets_reduce):
  for i in range(len(index_numpy)):
    d = index_numpy[i,0]
    t = index_numpy[i,1]
    # print(d)
    # print(t)

    drug = df_drugs_reduce[d:d + 1]
    # Reset the index values to the second dataframe appends properly
    drug = drug.reset_index(drop=True)
    # drop=True option avoids adding new index column with old index values
    # print(drug)

    target = df_targets_reduce[t:t + 1]
    # Reset the index values to the second dataframe appends properly
    target = target.reset_index(drop=True)
    # target=True option avoids adding new index column with old index values
    # print(target)

    horizontal_stack = pd.concat([drug, target], axis=1)
    # print(horizontal_stack)
    if i == 0:
      vertical_stack_dataset = horizontal_stack
      # print(vertical_stack_dataset)
    else:
      # Stack the DataFrames on top of each other
      vertical_stack_dataset = pd.concat([vertical_stack_dataset, horizontal_stack], axis=0)
      # print(vertical_stack_dataset)
  return vertical_stack_dataset

minority_train = mergeDrugTarget(minority_train_index_numpy,df_drugs_reduce,df_targets_reduce)
minority_train.columns = range(minority_train.shape[1])
print(minority_train)
#========================================================
# write minority to csv:
minority_train.to_csv("minority_train.csv", index=False)
#=================================================================
# write maority class index to csv:
df_majority_train_index = df_train_majority[['drug_id','target_id']]
print(df_majority_train_index)
df_majority_train_index.to_csv("df_majority_train_index.csv",index = False)
# ....
#==============> ****go to clustringMajority.py****


