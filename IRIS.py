# Import libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score,roc_curve
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore")

# 1)Problem Statement
#Predict the species of flower

# 2)Data Gathering
df=pd.read_csv("Iris.csv")
df

# 3)EDA(Exploratory Data Analysis)
df.info()

df.nunique()

df["Species"].value_counts()

# 4)Feature Engineering
 # Id column having more unique values so we want to remove that column as in prediction of model 
 # It is having no contribution

df.drop("Id",axis=1,inplace=True)

# 5)Feature selection
sns.pairplot(df,hue="Species")

# 6)Model Training
#seperate dependent(x) & independent(y) variables

x=df.drop("Species",axis=1)
y=df["Species"]

#Spliting dataset into training & testing

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

y_train.value_counts()

# A) Build the model for Logistic regression
#create instance & fit the model
lr_clf=LogisticRegression()
lr_clf.fit(x_train,y_train)

# 7)Model Evaluation for Logistic Regression
#Model evaluation for training data

y_pred_train=lr_clf.predict(x_train)

cnf_matrix=confusion_matrix(y_train,y_pred_train)
print("Confusion Matrix for training:\n",cnf_matrix)

accuracy=accuracy_score(y_train,y_pred_train)
print("Accuracy score for training:\n",accuracy)

clf_report=classification_report(y_train,y_pred_train)
print("Classification report for training:\n",clf_report)

#model evaluation for testing data
y_pred_test=lr_clf.predict(x_test)

cnf_matrix=confusion_matrix(y_test,y_pred_test)
print("Confusion Matrix for testing:\n",cnf_matrix)

accuracy=accuracy_score(y_test,y_pred_test)
print("Accuracy score for testing:\n",accuracy)

clf_report=classification_report(y_test,y_pred_test)
print("Classification report for testing:\n",clf_report)

# B) Build the model for KNN algorithm
#create instance & fit the model
knn_clf=KNeighborsClassifier()
knn_clf.fit(x_train,y_train)

# Model evaluation for KNN Alogirthm
#Model evaluation for training data

y_pred_train=knn_clf.predict(x_train)

cnf_matrix=confusion_matrix(y_train,y_pred_train)
print("Confusion Matrix for training:\n",cnf_matrix)

accuracy=accuracy_score(y_train,y_pred_train)
print("Accuracy score for training:\n",accuracy)

clf_report=classification_report(y_train,y_pred_train)
print("Classification report for training:\n",clf_report)

#model evaluation for testing data
y_pred_test=knn_clf.predict(x_test)

cnf_matrix=confusion_matrix(y_test,y_pred_test)
print("Confusion Matrix for testing:\n",cnf_matrix)

accuracy=accuracy_score(y_test,y_pred_test)
print("Accuracy score for testing:\n",accuracy)

clf_report=classification_report(y_test,y_pred_test)
print("Classification report for testing:\n",clf_report)

# Hyperparameter tuning for KNN algorithm

hyperparameters={"n_neighbors":np.arange(3,20),
                "p":[1,2]}
gscv_clf=GridSearchCV(knn_clf,hyperparameters,cv=5)
gscv_clf.fit(x_train,y_train)
gscv_clf.best_estimator_
#gscv_clf.best_params_

#create model instance for new hyperparameters
#knn_clf=KNeighborsClassifier(n_neighbors=6,p=2)
knn_clf=gscv_clf.best_estimator_
knn_clf.fit(x_train,y_train)

# Model evaluation for hyperparameter tuned KNN Alogirthm
#Model evaluation for training data
y_pred_train=knn_clf.predict(x_train)

cnf_matrix=confusion_matrix(y_train,y_pred_train)
print("Confusion Matrix for training:\n",cnf_matrix)

accuracy=accuracy_score(y_train,y_pred_train)
print("Accuracy score for training:\n",accuracy)

clf_report=classification_report(y_train,y_pred_train)
print("Classification report for training:\n",clf_report)

#model evaluation for testing data
y_pred_test=knn_clf.predict(x_test)

cnf_matrix=confusion_matrix(y_test,y_pred_test)
print("Confusion Matrix for testing:\n",cnf_matrix)

accuracy=accuracy_score(y_test,y_pred_test)
print("Accuracy score for testing:\n",accuracy)

clf_report=classification_report(y_test,y_pred_test)
print("Classification report for testing:\n",clf_report)

# C) Build the model for Decision Tree algorithm
#create instance & fit the model
dt_clf=DecisionTreeClassifier()
dt_clf.fit(x_train,y_train)

# Model evaluation for Decision Tree Alogirthm
#Model evaluation for training data
y_pred_train=dt_clf.predict(x_train)

cnf_matrix=confusion_matrix(y_train,y_pred_train)
print("Confusion Matrix for training:\n",cnf_matrix)

accuracy=accuracy_score(y_train,y_pred_train)
print("Accuracy score for training:\n",accuracy)

clf_report=classification_report(y_train,y_pred_train)
print("Classification report for training:\n",clf_report)

#model evaluation for testing data
y_pred_test=dt_clf.predict(x_test)

cnf_matrix=confusion_matrix(y_test,y_pred_test)
print("Confusion Matrix for testing:\n",cnf_matrix)

accuracy=accuracy_score(y_test,y_pred_test)
print("Accuracy score for testing:\n",accuracy)

clf_report=classification_report(y_test,y_pred_test)
print("Classification report for testing:\n",clf_report)

# Hyperparameter tuning for Decision tree algorithm
dt_clf=DecisionTreeClassifier()

#Hyperparameters
#     criterion='gini',
#     splitter='best',
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     min_weight_fraction_leaf=0.0,
#     max_features=None,
#     random_state=None,
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     class_weight=None,
#     ccp_alpha=0.0,

hypermarameters={"criterion":['gini','entropy'],
    "max_depth":np.arange(3,8),
    "min_samples_split":np.arange(3,10),
    "min_samples_leaf":np.arange(2,13)}
gscv_dt_clf=GridSearchCV(dt_clf,hypermarameters,cv=5)
gscv_dt_clf.fit(x_train,y_train)
gscv_dt_clf.best_estimator_

#create model instance for new hyperparameters
dt_clf=gscv_dt_clf.best_estimator_
dt_clf.fit(x_train,y_train)
# Model evaluation for hyperparameter tuned Decision Tree Alogirthm
#Model evaluation for training data
y_pred_train=dt_clf.predict(x_train)

cnf_matrix=confusion_matrix(y_train,y_pred_train)
print("Confusion Matrix for training:\n",cnf_matrix)

accuracy=accuracy_score(y_train,y_pred_train)
print("Accuracy score for training:\n",accuracy)

clf_report=classification_report(y_train,y_pred_train)
print("Classification report for training:\n",clf_report)

#model evaluation for testing data
y_pred_test=dt_clf.predict(x_test)

cnf_matrix=confusion_matrix(y_test,y_pred_test)
print("Confusion Matrix for testing:\n",cnf_matrix)

accuracy=accuracy_score(y_test,y_pred_test)
print("Accuracy score for testing:\n",accuracy)

clf_report=classification_report(y_test,y_pred_test)
print("Classification report for testing:\n",clf_report)

# d) Build the model for Random Forest algorithm
#create instance & fit the model
rf_model=RandomForestClassifier()
rf_model.fit(x_train,y_train)

#Model evaluation for training data
y_pred_train=rf_model.predict(x_train)

cnf_matrix=confusion_matrix(y_train,y_pred_train)
print("Confusion Matrix for training:\n",cnf_matrix)

accuracy=accuracy_score(y_train,y_pred_train)
print("Accuracy score for training:\n",accuracy)

clf_report=classification_report(y_train,y_pred_train)
print("Classification report for training:\n",clf_report)

#model evaluation for testing data

y_pred_test=rf_model.predict(x_test)

cnf_matrix=confusion_matrix(y_test,y_pred_test)
print("Confusion Matrix for testing:\n",cnf_matrix)

accuracy=accuracy_score(y_test,y_pred_test)
print("Accuracy score for testing:\n",accuracy)

clf_report=classification_report(y_test,y_pred_test)
print("Classification report for testing:\n",clf_report)

# Hyperparameter tuning for Random Forest algorithm

rf_model=RandomForestClassifier()

# Hyperparameters=
#     n_estimators=100,
#     criterion='gini',
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     min_weight_fraction_leaf=0.0,
#     max_features='sqrt',
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     bootstrap=True,
#     oob_score=False,
#     n_jobs=None,
#     random_state=None,
#     verbose=0,
#     warm_start=False,
#     class_weight=None,
#     ccp_alpha=0.0,
#     max_samples=None,

hyperparameters={"n_estimators":np.arange(5,100),
                "criterion":['gini','entropy'],
                "max_depth":np.arange(3,8),
                "min_samples_split":(2,6),
                "min_samples_leaf":np.arange(1,5),
                 "max_features":['sqrt'],
                "random_state":[11]}
rscv_rf_model=RandomizedSearchCV(rf_model,hyperparameters,cv=5)
rscv_rf_model.fit(x_train,y_train)
rscv_rf_model.best_estimator_

# Model evaluation for hyperparameter tuned Random Forest Alogirthm
#create model instance for new hyperparameters
rf_model=rscv_rf_model.best_estimator_
rf_model.fit(x_train,y_train)

#Model evaluation for training data
y_pred_train=rf_model.predict(x_train)

cnf_matrix=confusion_matrix(y_train,y_pred_train)
print("Confusion Matrix for training:\n",cnf_matrix)

accuracy=accuracy_score(y_train,y_pred_train)
print("Accuracy score for training:\n",accuracy)

clf_report=classification_report(y_train,y_pred_train)
print("Classification report for training:\n",clf_report)

#model evaluation for testing data

y_pred_test=rf_model.predict(x_test)

cnf_matrix=confusion_matrix(y_test,y_pred_test)
print("Confusion Matrix for testing:\n",cnf_matrix)

accuracy=accuracy_score(y_test,y_pred_test)
print("Accuracy score for testing:\n",accuracy)

clf_report=classification_report(y_test,y_pred_test)
print("Classification report for testing:\n",clf_report)

# Algorithm Name                               Training Accuracy                   Testing Accuracy
# 1)Logistic Regression                           0.975                                 0.966
# 2)KNN Alogorithm                                0.966                                 1.0
# 3)KNN with Hyperparameter tuning                0.966                                 0.966
# 4)Decision Tree                                 1.0                                   0.966
# 5)Decision Tree with Hypeparameter tuning       0.983                                 0.966
# 6)Random Forest                                 1.0                                   0.966
# 7)Random Forest with Hyperparameter tuning      0.975                                 0.933

# predict model on single row
x_test

x_test.iloc[4]

SepalLengthCm    = 4.4
SepalWidthCm     = 3.2
PetalLengthCm    = 1.3
PetalWidthCm     = 0.2
test_array=np.array([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
test_array

#Save the model by preparing pickle file for the model
import pickle
with open("knn.pkl","wb")as f:
    pickle.dump(knn_clf,f)

#Here label encoded values are not present so we create json for column values only
    
#Load the model 
with open("knn.pkl","rb") as f:
    load_model=pickle.load(f)

load_model.predict([[4.4,3.2,1.3,0.2]])

