# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 12:05:04 2021

@author: doguilmak

dataset: https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations

"""
#%%
# 1. Importing Libraries

import time
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.metrics import plot_confusion_matrix
warnings.filterwarnings("ignore")

#%%
# 2. Data Preprocessing

# 2.1. Importing Data
start = time.time()
df = pd.read_csv('diagnosis.csv', header=None)

# 2.2. Looking For Anomalies
print(df.info()) # Looking for the missing values
print(df.describe().T)
print(df.isnull().sum())
print("Number of NaN values: ", df.isnull().sum().sum())

# 3.3. Label Encoding
from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)
print("data:\n", df)

# 2.4. DataFrame Slice - Determination of Dependent and Independent Variables
x = df.iloc[:, 0:4].values 
y = df.iloc[:, 7].values # Nephritis of renal pelvis origin

# 2.5. Train - Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# 2.6. Scaling Datas
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#%%
# Logistic Regression

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
y_pred = logr.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Confusion Matrix - Accuracy Score
cm = confusion_matrix(y_test, y_pred)
print('\nLogistic Regression Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")

# Plotting Confusion Matrix
plot_confusion_matrix(logr, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Logistic Regression Classifier')  
plt.show()

#%%
# K-NN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

# Confusion Matrix - Accuracy Score
cm = confusion_matrix(y_test,y_pred)
print('\nKNN Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
print(f"K-NN Score: {knn.score(X_test, y_test)}")

# Plotting Confusion Matrix
plot_confusion_matrix(knn, X_test, y_test)
plt.title('K-NN Classifier')  
plt.show()

#%%
# SVM

from sklearn.svm import SVC
svc = SVC(kernel='sigmoid', C=4, gamma=0.1) # linear-rbf-sigmoid-precomputed-callable
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

# Confusion Matrix - Accuracy Score
cm = confusion_matrix(y_test,y_pred)
print('\nSVC Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")

# Plotting Confusion Matrix
plot_confusion_matrix(svc, X_test, y_test, cmap=plt.cm.Blues)
plt.title('SVC Classifier')  
plt.show()

#%%
# Desicion Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dtc = DecisionTreeClassifier(criterion = 'entropy') # gini(from Information Gain JPEG Pi*Pi) - entropy
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test) # Tahmin işlemi

# Confusion Matrix - Accuracy Score
cm = confusion_matrix(y_test, y_pred)
print('\nDecision Tree Classifier Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
tree.plot_tree(dtc) 

# Plotting Confusion Matrix
plot_confusion_matrix(dtc, X_test, y_test)
plt.title('Decision Tree Classifier')
plt.show()

#%%
# Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test) # Tahmin işlemi

# Confusion Matrix - Accuracy Score
cm = confusion_matrix(y_test, y_pred)
print('\nRandom Forest Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")

# Plotting Confusion Matrix
plot_confusion_matrix(dtc, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Random Forest Classifier')
plt.show()

#%%
# ROC , TPR, FPR Values of Logistic Regression

from sklearn import metrics
print("\nPredict Probability")
y_proba = logr.predict_proba(X_test) # Elemanları test eder ve provaları verir
print("Predict probability:\n", y_proba) # 1.Sütun: Sepal Length olasılıkları, 2.Sütun: Sepal width olasılıkları, 3.Sütun: Petal Length

y_pred_proba = logr.predict_proba(X_test)[::,1]
fpr , tpr , thold = metrics.roc_curve(y_test, y_proba[:, 1])
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print('AUC Score:\n', auc)
print("False Positive Rate:\n", fpr)
print("True Positive Rate:\n", tpr)

font1 = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 10,
        }
font2 = {'family': 'serif',
         'color': 'black',
         'weight': 'normal',
         'size': 15,
         }

lw = 1

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.plot(fpr, tpr, color='red', linestyle='-', marker='o', markerfacecolor='black', markersize=5)
plt.title("ROC", fontdict=font2)
plt.xlabel("False Positive Rate", fontdict=font1)
plt.ylabel("True Positive Rate", fontdict=font1)
plt.show()

#%%
# K-Fold Cross Validation

from sklearn.model_selection import cross_val_score

success = cross_val_score(estimator = svc, X=X_train, y=y_train, cv = 4)
print("\nK-Fold Cross Validation:")
print("Success Mean:\n", success.mean())
print("Success Standard Deviation:\n", success.std())

#%%
# Grid Search

from sklearn.model_selection import GridSearchCV
p = [{'C':[1,2,3,4,5],'kernel':['linear'], 'gamma':[1,0.5,0.1,0.01,0.001]},
     {'C':[1,2,3,4,5] , 'kernel':['rbf'], 'gamma':[1,0.5,0.1,0.01,0.001]},
     {'C':[1,2,3,4,5] , 'kernel':['sigmoid'], 'gamma':[1,0.5,0.1,0.01,0.001]},
     {'C':[1,2,3,4,5] , 'kernel':['callable'], 'gamma':[1,0.5,0.1,0.01,0.001]}]


gs = GridSearchCV(estimator= svc,
                  param_grid=p,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1)

grid_search = gs.fit(X_train, y_train)
best_result = grid_search.best_score_
best_parameters = grid_search.best_params_
print("\nGrid Search")
print("Best result:\n", best_result)
print("Best parameters:\n", best_parameters)


#%%
# Saving Model
"""
import pickle
file = "logr.save"
pickle.dump(logr, open(file, 'wb'))

downloaded_data = pickle.load(open(file, 'rb'))
print(downloaded_data.predict(X_test))
"""

end = time.time()
cal_time = end - start
print("\nTook {} seconds to classificate objects.".format(cal_time))
