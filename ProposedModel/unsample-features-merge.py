
#   *****run this code for each cluster******
#   1:find drugs in each cluster
#   2:undersampling of each cluster majority class
#   3:merge feature of drug and target in each cluster majority
#   4:construct lables
#   5:Combine minority class with downsampled majority class
#=================================================================
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.utils import resample
#=========================================
df_minority = pd.read_csv('files/minority_train.csv')
print(df_minority)

df_cluster_drugs = pd.read_csv('files/drugs_cluster.csv')
df_cluster_drugs

#Now, let see how much drugs we have in each cluster:
for i in range(11):
    len_drugs = df_cluster_drugs[df_cluster_drugs['Cluster'] == i].shape[0]
    print('drugs in Cluster ' + str(i) + ' -> ', len_drugs)

df_majority_train_index = pd.read_csv('files/df_majority_train_index.csv')
print(df_majority_train_index)
#=========================================
# run for each cluster:
#change k for each cluster:
# for all cluster with set k:
# k is cluster number:k = 0,1,2,3,4,5,6,7,8,9,10
k = 10
drugs_cluster_k = df_cluster_drugs[df_cluster_drugs['Cluster'] == k]
# print(drugs_cluster_k)
drugs_id_cl_k = drugs_cluster_k['drug_id'].to_numpy()
# print(drugs_id_cl_k)
len_drugs = len(drugs_id_cl_k)
print("number of drugs in cluster {} = {}" .format(str(k),str(len_drugs)))
print(drugs_id_cl_k)
for i in range(len(drugs_id_cl_k)):
  if i == 0:
      majority_index_zero_k = df_majority_train_index[df_majority_train_index['drug_id'] == drugs_id_cl_k[i]]
  else:
      majority_index_zero_k = pd.concat([majority_index_zero_k,
                                         df_majority_train_index[df_majority_train_index['drug_id'] == drugs_id_cl_k[i]]], axis=0)

print(majority_index_zero_k)
#=====================================================================
# Downsample from each cluster of majority class:
majority_index_zero_downsampledCluster_k = resample(majority_index_zero_k,
                                   replace=False,  # sample without replacement
                                   n_samples=len(df_minority),  # to match minority class
                                   random_state=123)  # reproducible results

print(majority_index_zero_downsampledCluster_k)
#==============================================================
# read reduced dimention of drugs and targets:
df_drugs_reduce = pd.read_csv('files/reducedFeaturesDrugs.csv')
print(df_drugs_reduce)
df_targets_reduce = pd.read_csv('files/reducedFeaturesTargets.csv')
print(df_targets_reduce)
#===========================================================
# merge:
def mergeDrugTarget(index_numpy,df_drugs_reduce,df_targets_reduce):
  for i in range(len(index_numpy)):
    # print("i:"+ str(i))
    d = index_numpy[i,0]
    t = index_numpy[i,1]
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
#===============================================
majority_index = majority_index_zero_downsampledCluster_k.to_numpy()
print(majority_index)
print(len(majority_index))
majority_cluster_k = mergeDrugTarget(majority_index,df_drugs_reduce,df_targets_reduce)
majority_cluster_k.columns = range(majority_cluster_k.shape[1])
majority_cluster_k = majority_cluster_k.reset_index(drop=True)
print(majority_cluster_k)
#====================================================================
majority_cluster_k.columns = range(majority_cluster_k.shape[1])
df_minority.columns = range(df_minority.shape[1])
# construct lables for minority and clusters of majority class:
def makeLable(len_df,lable):
  if lable == 0:
    lables = np.zeros(shape=(len_df,1))
    # print(lables)
    # print(type(lables))
  elif lable == 1:
    lables = np.ones(shape=(len_df,1))
  return lables

lables = makeLable(len(df_minority),1)
df_minority['lable'] = lables
print(df_minority)

lables = makeLable(len(majority_cluster_k),0)
majority_cluster_k['lable'] = lables
print(majority_cluster_k)
#=======================================
# Combine minority class with downsampled majority class
# print(type(majority_cluster_k))
# print(type(df_minority))
# print(df_minority)
# print(majority_cluster_k)

df_downsampled_k = pd.concat([majority_cluster_k, df_minority])
df_downsampled_k = df_downsampled_k.reset_index(drop=True)
print(df_downsampled_k)
# Display new class counts

# write subDate of maority and downsampled majority class to csv:
df_downsampled_k.to_csv("clusterOFtraining/df_downsampled_10.csv",index = False)