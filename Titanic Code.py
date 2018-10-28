#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 16:13:39 2018

@author: Dengyuan
"""
#import syy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib inline

#1: Loading Datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


data_temp = train[['Ticket', 'Cabin']]
data_temp.head()

#2: Check Datasets
train.shape
train.info()

#3: Find the null values
train.isnull().sum()

#4:
test.shape
test.info()

#5: Calculate survivial rate
survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]

print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))
print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))
print ("Total: %i"%len(train))

#6: Analyze Pclass data
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
sns.barplot(x='Pclass', y='Survived', ci=None, data=train)

#7: Analyze Sex data
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
sns.barplot(x='Sex', y='Survived', ci=None, data=train)

#8: Analyze Embarked data
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', ci=None, data=train)

#9: Analyze Parent-Children data
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
sns.barplot(x='Parch', y='Survived', ci=None, data=train)

#10: Analyze Sibling data
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()
sns.barplot(x='SibSp', y='Survived', ci=None, data=train) 

#11: Correlatiing Features
plt.figure(figsize=(15,5))
colormap = sns.diverging_palette(220, 10, as_cmap = True)
sns.heatmap(train.drop('PassengerId',axis=1).corr(), cmap=colormap, vmax=0.6, square=True, annot=True)

#12 Combine train and test Datasets
train_test_data = [train, test] 

#13 Complete Datasets
for dataset in train_test_data:
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

#14: Extract titles from Name column. 
for dataset in train_test_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

#15:
pd.crosstab(train['Title'], train['Sex'])

#16: Replace some less common titles with the name "Other"
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#17: Convert the categorical Title values into numeric form
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)    
    
#18: Convert Sex data
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)    
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#19: Categorizes age into 5 different age range   
train['AgeBand'] = pd.cut(train['Age'], 5)

#20: Show age bar chart
sns.barplot(x='AgeBand', y='Survived', ci=None, data=train)

#21: Convert Agebands
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    
#22: Replace missing Fare values with the median of Fare
for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median()) 

#23: Create 4 FareBands
train['FareBand'] = pd.qcut(train['Fare'], 4)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())

#24: Map Fare according to FareBands
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

#25: Combining SibSp & Parch feature, we create a new feature named FamilySize
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

#26: Create a new feature named IsAlone
for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

#27: Drop unnecessary columns/features 
# Option 1: Training Data with Title feature
#features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
# Option 2: Training Data without Title feature
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize', 'Title']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)

#28: Define training and testing set
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop("PassengerId", axis=1).copy()

X_train.head()
y_train.head()
X_train.shape, y_train.shape, X_test.shape

#29: Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#30: Logistic Regression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
print (str(acc_log_reg) + ' percent')

#31: Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print (acc_decision_tree)

#32: Random Forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print (acc_random_forest)
    
#33: Compare Models using Accuracy Scores
models = pd.DataFrame({
    'Model': ['Logistic Regression',   
              'Decision Tree', 
              'Random Forest'],
    
    'Accuracy Score': [acc_log_reg, 
              acc_decision_tree, 
              acc_random_forest]
    })

models.sort_values(by='Accuracy Score', ascending=False)


#34: Get Confusion Matrics for Random Forest
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision=2)
true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']
# Compute confusion matrix

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage for Random Forest')
true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)

#35: Get Confusion Matrics for Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree_set = clf.predict(X_train)
cnf_matrix = confusion_matrix(y_train, y_pred_decision_tree_set)

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage for Decision Tree')
true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)

#36: Get Confusion Matrix for Logistic Regression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg_set = clf.predict(X_train)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree_set = clf.predict(X_train)
cnf_matrix = confusion_matrix(y_train, y_pred_log_reg_set)

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage for Logistic Regression')
true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)

X_train.head()
