# 2:
#   2.1:clustring majority class
#=====================================
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
#==============================================================
df_index = pd.read_csv('files/df_majority_train_index.csv')
print(df_index)
#==============================================================
targets_id = np.unique(df_index['target_id'])
print(targets_id)
drugs_id = np.unique(df_index['drug_id'])
print(drugs_id)
#==============================================================
def targetListForDrugs(drugs, drugs_data):
    # drugs = a list of drugs IDs
    # drugs_data = index of majority class
    drugs_targets_list = []
    for drug in drugs:
        drugs_targets_list.append(
            str(list(drugs_data[drugs_data['drug_id'] == drug]['target_id'])).split('[')[1].split(']')[0])
    return drugs_targets_list

drugs_targets_list = targetListForDrugs(drugs_id, df_index)
# print('targets list for', len(drugs_targets_list), ' drugs')
# print('A list of first 2 drugs favourite targets: \n', drugs_targets_list[:2])
# ===========>time of run:2min
#==============================================================
def prepSparseMatrix(list_of_str):
  # list_of_str = A list, which contain strings of drugs interaction targets separate by comma ",".
  # It will return us sparse matrix and feature names on which sparse matrix is defined
  # i.e. name of target in the same order as the column of sparse matrix
  cv = CountVectorizer(token_pattern = r'[^\,\ ]+', lowercase = False)
  sparseMatrix = cv.fit_transform(list_of_str)
  return sparseMatrix.toarray(), cv.get_feature_names()
sparseMatrix, feature_names = prepSparseMatrix(drugs_targets_list)
df_sparseMatrix = pd.DataFrame(sparseMatrix, index = drugs_id, columns = feature_names)
#==============================================================
class elbowMethod():
  def __init__(self, sparseMatrix):
      self.sparseMatrix = sparseMatrix
      self.wcss = list()
      self.differences = list()
  def run(self, init, upto, max_iterations = 100):
      for i in range(init, upto + 1):
        kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter = max_iterations, n_init = 10, random_state = 1)
        kmeans.fit(self.sparseMatrix)
        self.wcss.append(kmeans.inertia_)
      self.differences = list()
      for i in range(len(self.wcss)-1):
        self.differences.append(self.wcss[i] - self.wcss[i+1])
  def showPlot(self, boundary = 500, upto_cluster = None):
      if upto_cluster is None:
        WCSS = self.wcss
        DIFF = self.differences
      else:
        WCSS = self.wcss[:upto_cluster]
        DIFF = self.differences[:upto_cluster - 1]
      plt.figure(figsize=(15, 6))
      plt.subplot(121).set_title('Elbow Method Graph')
      plt.plot(range(1, len(WCSS) + 1), WCSS)
      plt.grid(b = True)
      plt.subplot(122).set_title('Differences in Each Two Consective Clusters')
      len_differences = len(DIFF)
      X_differences = range(1, len_differences + 1)
      plt.plot(X_differences, DIFF)
      plt.plot(X_differences, np.ones(len_differences)*boundary, 'r')
      plt.plot(X_differences, np.ones(len_differences)*(-boundary), 'r')
      plt.grid()
      plt.show()
# elbow_method = elbowMethod(sparseMatrix)
# elbow_method.run(1, 9)
# elbow_method.showPlot(boundary = 600)
# elbow_method.run(10, 20)
# elbow_method.showPlot(boundary = 600)
#========>so the number of clusters is 10 for majority train set
# ===========>time of run:12min
#==============================================================
kmeans = KMeans(n_clusters=11, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
clusters = kmeans.fit_predict(sparseMatrix)



