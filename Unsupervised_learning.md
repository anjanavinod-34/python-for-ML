## Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns

### Loading the test data
# Loading the test data

url = 'https://query.data.world/s/ksxft7lhmbxpihskwsngwhpuul6lye'
col_names = ['target','F1R', 'F1S', 'F2R', 'F2S', 'F3R', 'F3S', 'F4R', 'F4S', 'F5R','F5S','F6R','F6S','F7R','F7S','F8R','F8S','F9R','F9S','F10R',
    'F10S',  'F11R','F11S','F12R','F12S','F13R','F13S','F14R','F14S','F15R','F15S','F16R','F16S','F17R','F17S','F18R','F18S','F19R','F19S',   'F20R',
    'F20S','F21R','F21S','F22R','F22S']
spectf_df_test= pd.read_table(url,sep=',',names=col_names)

spectf_df_test
# Loading the Train data

url = 'https://query.data.world/s/cuqtpuoewpxysusrt5z4igihjah4xo'
col_names = ['target','F1R', 'F1S', 'F2R', 'F2S', 'F3R', 'F3S', 'F4R', 'F4S', 'F5R','F5S','F6R','F6S','F7R','F7S','F8R','F8S','F9R','F9S','F10R',
    'F10S',  'F11R','F11S','F12R','F12S','F13R','F13S','F14R','F14S','F15R','F15S','F16R','F16S','F17R','F17S','F18R','F18S','F19R','F19S',   'F20R',
    'F20S','F21R','F21S','F22R','F22S']
spectf_df= pd.read_table(url,sep=',',names=col_names)

spectf_df
spectf_df.info()
spectf_df.describe()
# Checking whether the dataset is balanced or not  

#### This code creates a count plot of the 'target' variable in the spectf_df dataset using seaborn.
#Train Data balance check
sns.countplot(x='target',data=spectf_df)
# Test data balance check
sns.countplot(x='target',data=spectf_df_test)
corr_df=spectf_df.corr()
plt.figure(figsize=(18,12))
sns.heatmap(corr_df)

# Split labels and features
traget=spectf_df['target']
spectf_df.drop('target', axis=1, inplace=True)
spectf_df
traget_test=spectf_df_test['target']
spectf_df_test.drop('target', axis=1, inplace=True)
spectf_df_test
Steps of PCA
1. Standarization
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()#instantiate
 
# compute the mean and standard which will be used in the next command

spect_df=scaler.fit_transform(spectf_df)# fit and transform can be applied together and I leave that for simple exercise
# we can check the minimum and maximum of the scaled features which we expect to be 0 and 1

spect_df_test=scaler.fit_transform(spectf_df_test)

traget
# Time taken to train

import time
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
start=time.time()


logreg.fit(spect_df,traget)
end=time.time()


traintime=end-start
print('training time for logisticreggression(logreg) is:  ',traintime)


y_pred_class=logreg.predict(spect_df_test)
from sklearn import metrics
metrics.accuracy_score(y_pred_class,traget_test)
spect_df
# Apply PCA to logreg model
from sklearn.decomposition import PCA

pca=PCA()
pca.fit(spect_df)
spect_df
# Min number of components we need
pca.explained_variance_ratio_[:15].sum()
pca.explained_variance_ratio_
pca.explained_variance_ratio_[:15].sum()
# 1st PC

pca.explained_variance_ratio_[:2].sum()
np.cumsum(pca.explained_variance_ratio_)
# Relationship b/w PC and variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.annotate('15', xy=(15,.90))

spect_df
x_pca=PCA(n_components=15)
spect_df_pca=x_pca.fit_transform(spect_df)
spect_df_pca_test=x_pca.fit_transform(spect_df_test)
spect_df_pca
pca_corr=pd.DataFrame(spect_df_pca).corr()
plt.figure(figsize=(18,12))
sns.heatmap(pca_corr,annot=True)
cols=['F1R', 'F1S', 'F2R', 'F2S', 'F3R', 'F3S', 'F4R', 'F4S', 'F5R','F5S','F6R','F6S','F7R','F7S','F8R','F8S','F9R','F9S','F10R',
    'F10S',  'F11R','F11S','F12R','F12S','F13R','F13S','F14R','F14S','F15R','F15S','F16R','F16S','F17R','F17S','F18R','F18S','F19R','F19S',   'F20R',
    'F20S','F21R','F21S','F22R','F22S']


pca_Df=pd.DataFrame(x_pca.components_, columns=cols)
pca_Df

plt.figure(figsize=(18,12))
sns.heatmap(pca_Df,cmap='RdYlGn', annot=True)
## Your task is to create the logreg model and put PCs as features and check the accuracy and how long it takes to train
# Time taken to train

import time
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
start=time.time()


logreg.fit(spect_df_pca,traget)
end=time.time()


traintime=end-start
print('training time for logreg is:  ',traintime)

y_pred_class1=logreg.predict(spect_df_pca_test)
from sklearn import metrics
metrics.accuracy_score(y_pred_class1,traget_test)



