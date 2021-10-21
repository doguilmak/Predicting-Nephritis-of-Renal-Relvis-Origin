# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 12:44:09 2021

@author: doguilmak

dataset: https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations

PCA using Python: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

"""
#%%
#  1. Libraries

import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

#%%
# 2. Data Preprocessing

# 2.1. Uploading data
start = time.time()
df = pd.read_csv('diagnosis.csv', names=['Temperature', 
                                         'Occurrence_of_nausea', 
                                         'Lumbar_pain', 'Urine_pushing', 
                                         'Micturition_pains', 
                                         'Burning_of_urethra', 
                                         'Inflammation_of_urinary_bladder', 
                                         'Nephritis_of_renal_pelvis_origin'])

# 2.2. Looking For Anomalies
print(df.info()) # Looking for the missing values
print(df.describe().T)
print(df.isnull().sum())
print("Number of NaN values: ", df.isnull().sum().sum())

# 3.3. Label Encoding
from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)
print("data:\n", df)

# 2.4. Seperate the Data Depending on Dependent and Independent Variableles
y = df["Nephritis_of_renal_pelvis_origin"]
X = df.drop("Nephritis_of_renal_pelvis_origin", axis = 1)

# 2.5. Split as Train and Test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 2.6. Scaling Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%
# 3.1. PCA

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)  # 2 dimensional

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

import matplotlib.pyplot as plt
principalDf = pd.DataFrame(data = X_train2,
              columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['Nephritis_of_renal_pelvis_origin']]], axis = 1)

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(1, 1, 1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Nephritis_of_renal_pelvis_origin'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 20)
ax.legend(targets)
ax.grid()

# 3.2. LR Transform Before PCA
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# 3.3. LR After PCA Transform
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

# 3.4. Predictions
y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# 3.5. Actual / Without PCA 
print('Actual / Without PCA')
cm1 = confusion_matrix(y_test, y_pred)
print(cm1)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}\n")

# 3.6. Actual / Result after PCA
print("Actual / With PCA")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)
print(f"Accuracy score: {accuracy_score(y_test, y_pred2)}\n")

# 3.7. After PCA / Before PCA
print('Without PCA and with PCA')
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)
print(f"Accuracy score: {accuracy_score(y_pred, y_pred2)}\n")

#%%
# 4. LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 1)  # 1 dimensional

X_train_lda = lda.fit_transform(X_train, y_train)  # In order for LDA to learn, the y_train parameter is entered.
X_test_lda = lda.transform(X_test)

# 4.1. After LDA Transform
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda, y_train)

# 4.2. Predict LDA Datas
y_pred_lda = classifier_lda.predict(X_test_lda)

# 4.3. After LDA / Actual
print('LDA and Actual')
cm4 = confusion_matrix(y_pred, y_pred_lda)
print(cm4)
print(f"Accuracy score: {accuracy_score(y_pred, y_pred_lda)}\n")

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
