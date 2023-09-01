# 5:
#==============================================
# import Required packages
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from numpy import mean
from numpy import std
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from imblearn.metrics import classification_report_imbalanced
from imblearn.metrics import geometric_mean_score,sensitivity_score,specificity_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils import resample
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
#===========================================================
df_downsampled = pd.read_csv('files/df_downsampled.csv')
print(df_downsampled)
X = df_downsampled.drop(columns = 'lable')
y = df_downsampled.lable
# print(X)
# print(y)
#split train/validation on clusters:
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15,stratify=y)
# print(y_val.value_counts())
#===========================================================
# add models:
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(probability=True)))
models.append(('RF', RandomForestClassifier()))
# print(type(models))
#======================================================
# evaluate each model:
acc_result = []
results = []
names = []
for name, model in models:
  kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
  cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  acc_result.append(cv_results.mean())
  print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()

#====================================================================
# calculate weihgt of clfs:
accuracy = []
dict_w_clfs = {}

for result in results:
  accuracy.append(result.mean())
print(accuracy)
sum_acc = sum(accuracy)
print(sum_acc)
i = 0
for acc in accuracy:
  dict_w_clfs[names[i]] =  acc / sum_acc
  i = i + 1
print(dict_w_clfs)
#=====================================================
# update weight of models:
list_prediction = []
for name, model in models:
  print(name)
  model.fit(X_train, y_train)
  prediction = model.predict(X_val)
  # print(type(prediction))
  # print(type(y_val))
  list_prediction.append((name, prediction))
  # print(list_prediction)
  print('accuracy: %.3f' % accuracy_score(y_val,prediction))
  probs = model.predict_proba(X_val)
  probs = probs[:, 1]
  # calculate AUC
  AUC_models = roc_auc_score(y_val, probs)
  print('AUC: %.3f' % AUC_models)
#==============
# update weights:
y_validation = y_val.to_numpy()
print(y_validation)
# print(type(dict_w_clfs))
for name, predict in list_prediction:
  for i in range(len(predict)):
    if predict[i] == y_validation[i]:
      reward = 0.01
      dict_w_clfs[name] = dict_w_clfs[name] + reward
      # print(reward)
    else:
      penalty = 0.01
      dict_w_clfs[name] = dict_w_clfs[name] - penalty
      # print(penalty)
print(dict_w_clfs)
weights = list(dict_w_clfs.values())
print(weights)
#==================================================================
# # remove weak model:
# print(type(weights))
indices = []
avg_w = (mean(weights))
print(avg_w)
max_w = max(weights)
print(max_w)
threshold = max_w-(max_w/avg_w+1)
print(threshold)
for weight in weights:
    if weight<threshold:
        index = weights.index(weight)
        indices.append(index)
# indices = [0,2,4,5,8]
print(indices)
models = [i for j, i in enumerate(models) if j not in indices]
weights = [i for j, i in enumerate(weights) if j not in indices]
print(models)
#====================================
# make ensemble(my model):
voting_clf_hard = VotingClassifier(models,voting='soft', weights=weights)
#================================