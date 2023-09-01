import pandas as pd
#==============================================
#preprocessing data : 1. convert txt files to csv files:
#===============================================
read_file = pd.read_csv (r'txtfiles/featureDrug.txt')
read_file.to_csv (r'dataWithoutHeader\\featureDrug.csv',index = False)
#drug feature vector normalize with min-max
drugs = pd.read_csv('dataWithoutHeader/featureDrug.csv', sep='\t', header = None)
print(drugs)
#=================================================
read_file = pd.read_csv (r'txtfiles/featureTarget.txt')
read_file.to_csv (r'dataWithoutHeader\\featureTarget.csv', index = False)
#target feature vector normalize with min-max
targets = pd.read_csv('dataWithoutHeader/featureTarget.csv', sep='\t', header = None)
print(targets)
#================================================
read_file = pd.read_csv (r'txtfiles/interactions.txt')
read_file.to_csv (r'interaction\\interactions.csv', index = False)
#intraction matrix : zero or one
interactions = pd.read_csv('interaction/interactions.csv', sep='\t', header = None)
print(interactions)
#=========================================
