import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

dia = pd.read_csv('C:/Users/ralph/Downloads/dia - diabetes_data_upload.csv')
dia.head()

dia.describe()

dia.dtypes

n_dia = LabelEncoder()
dia['Gender'] = n_dia.fit_transform(dia['Gender'].astype('str')) 
dia['Polyuria'] = n_dia.fit_transform(dia['Polyuria'].astype('str')) 
dia['Polydipsia'] = n_dia.fit_transform(dia['Polydipsia'].astype('str')) 
dia['sudden weight loss'] = n_dia.fit_transform(dia['sudden weight loss'].astype('str')) 
dia['weakness'] = n_dia.fit_transform(dia['weakness'].astype('str')) 
dia['Obesity'] = n_dia.fit_transform(dia['Obesity'].astype('str')) 
dia['class'] = n_dia.fit_transform(dia['class'].astype('str')) 
dia.head()

x = dia.iloc[:, :-1].values
y = dia.iloc[:, 7]
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 6 Algorithms for Base Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# 1st Algorithm: Decision Tree 
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
pred_dt = dt.predict(x_test)
print(y_test)
print(pred_dt)
print(metrics.accuracy_score(y_test, pred_dt)*100)
print(confusion_matrix(y_test, pred_dt))
print(classification_report(y_test, pred_dt))

# 2nd Algorithm: Gaussian NB
nb = GaussianNB()
nb.fit(x_train, y_train)
pred_nb = nb.predict(x_test)
print(y_test)
print(pred_nb)
print(metrics.accuracy_score(y_test, pred_nb)*100)
print(confusion_matrix(y_test, pred_nb))
print(classification_report(y_test, pred_nb))

# 3rd Algorithm: Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
pred_lr = lr.predict(x_test)
print(y_test)
print(pred_lr)
print(metrics.accuracy_score(y_test, pred_lr)*100)
print(confusion_matrix(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

# 4th Algorithm: K-Nearest Neighbours
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
pred_knn = knn.predict(x_test)
print(y_test)
print(pred_knn)
print(metrics.accuracy_score(y_test, pred_knn)*100)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))

# 5th Algorithm: Support Vector Machine
svm = SVC()
svm.fit(x_train, y_train)
pred_svm = svm.predict(x_test)
print(y_test)
print(pred_svm)
print(metrics.accuracy_score(y_test, pred_svm)*100)
print(confusion_matrix(y_test, pred_svm))
print(classification_report(y_test, pred_svm))

# 6th Algorithm: Neural Networks
nn = MLPClassifier(hidden_layer_sizes =(200, 200, 200), activation = 'relu', max_iter = 100)
nn.fit(x_train, y_train)
pred_nn = dt.predict(x_test)
print(y_test)
print(pred_nn)
print(metrics.accuracy_score(y_test, pred_nn)*100)
print(confusion_matrix(y_test, pred_nn))
print(classification_report(y_test, pred_nn))

# Additonal Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# 7th Algorithm: Random Forest
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
pred_rf = rf.predict(x_test)
print(y_test)
print(pred_svm)
print(metrics.accuracy_score(y_test, pred_rf)*100)
print(confusion_matrix(y_test, pred_rf))
print(classification_report(y_test, pred_rf))

# 8th Algorithm: Gradient Boosting Classifier
gb = GradientBoostingClassifier()
gb.fit(x_train, y_train)
pred_gb = gb.predict(x_test)
print(y_test)
print(pred_gb)
print(metrics.accuracy_score(y_test, pred_gb)*100)
print(confusion_matrix(y_test, pred_gb))
print(classification_report(y_test, pred_gb))

