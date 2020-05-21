#!/usr/bin/env python
# coding: utf-8

# In[206]:


import math
def euclidean_distance(row1, row2):
    return np.linalg.norm(row1-row2)
#def manhattan_distance(row1,row2):


# In[207]:


import numpy as np
import matplotlib.pyplot as plt
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# In[208]:


def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# In[209]:


import pandas as pd
from sklearn.utils import resample
def readcsv(filename):
    data = pd.read_csv(filename)
    corrmat = data.corr()
    corrfeatures = set()
    for i in range(len(corrmat.columns)):
        for j in range(i):
            if(abs(corrmat.iloc[i,j])>0.95):
                colname = corrmat.columns[i]
                corrfeatures.add(colname)
    for i in corrfeatures:
        #print("Entered")
        data = data.drop(i,axis = 1)
    """n_sample = data["class"].value_counts()
    #print(n_sample)
    data_majority = data[data["class"]==1]
    data_minority = data[data["class"]==0]
    data_minority_upsampled = resample(data_minority,replace=True,n_samples=3824,random_state=123)
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])
    print(data_upsampled["class"].value_counts())"""
    return(np.array(data))
dataset = readcsv("cat3.csv")
point = int(0.6*(len(dataset)))
valend = int(0.8*(len(dataset)))
train = dataset[0:point+1]
val=dataset[point+1:valend+1]
test = dataset[valend+1:]


# In[ ]:





# In[ ]:





# In[210]:


labels = list()
accuracies = list()
Specificity = list()
Sensitivity = list()
F1score = list()
precision = list()
recall = list()
for x in range(3,int(math.sqrt(len(train))/2)+1,1):
    labels.clear()
    for i in range(len(val)):
        #print(i)
        label = predict_classification(train,val[i],x)
        #print(label)
        labels.append(label)
    TN1=0
    TP1=0
    FP1=0
    FN1=0
    for i in range(len(labels)):
        if(val[i][-1]==labels[i]):
            if(labels[i]==0):
                TN1=TN1+1
            else: 
                TP1=TP1+1
        else:
            if(labels[i]==1):
                FP1=FP1+1
            else:
                FN1=FN1+1
    #print(x)
    #precision1 = TP1/(TP1 + FP1)
    #recall1 = TP1/(TP1+FN1)
    #precision.append(TP1/(TP1 + FP1))
    #recall.append(TP1/(TP1+FN1))
    if(TP1+TN1==0):
        accuracies.append(0)
    else:
        accuracies.append(((TP1+TN1)/(TN1+TP1+FN1+FP1)))
    #F1score.append(2*((precision1*recall1)/(precision1+recall1)))
    #Specificity.append((TN1/(TN1+FP1)))
    #Sensitivity.append((TP1/(TP1+FN1)))


# In[211]:


#print(Sensitivity)


# In[212]:


k=accuracies.index(max(accuracies))+3
print(k)


# In[213]:


#print(len(test))


# In[214]:


labels.clear()
for i in range(len(test)):
    print(i)
    """With the value of k found using validation set, running knn on test set"""
    label = predict_classification(train,test[i],k)
    #print(label)
    labels.append(label)


# In[229]:


TN=0
TP=0
FP=0
FN=0
for i in range(len(labels)):
    if(test[i][-1]==labels[i]):
        if(labels[i]==0):
            TN=TN+1
            TP=TP
        else: 
            TP=TP+1
    else:
        if(labels[i]==1):
            FP=FP+1
        else:
            FN=FN+1
if(TP==0):
    precision2=0
    recall2=0
else:
    recall2 = TP/(TP+FN)
    precision2 = TP/(TP + FP)
if(precision2==0 and recall2==0):
    f1score=0
else:
    f1score=2*((precision2*recall2)/(precision2+recall2))
if(TN+TP==0):
    accuracy=0
else:
    accuracy=((TN+TP)/(TN+TP+FP+FN))
print(accuracy)
print(f1score)


# In[230]:


fpr = list()
for i in Specificity:
    fpr.append(1-i)
#plt.scatter(Sensitivity,fpr)
#plt.plot(precision,recall)


# In[231]:


"""dataset = readcsv("cat4.csv")
point = int(0.6*(len(dataset)))
valend = int(0.8*(len(dataset)))
train = dataset[0:point+1]
val=dataset[point+1:valend+1]
test = dataset[valend+1:]"""


# In[232]:


#print(len(dataset))


# In[233]:


"""M = np.mean(dataset.T, axis=1)
#print(M)
# center columns by subtracting column means
C = dataset - M
#print(C)
# calculate covariance matrix of centered matrix
V = np.cov(C.T)
#print(V)
# eigendecomposition of covariance matrix
values, vectors = np.linalg.eig(V)
print(vectors)
print(values)
# project data
P = vectors.T.dot(C.T)
print(P.T)"""
from sklearn.preprocessing import StandardScaler
dataset=pd.read_csv("cat3.csv")
data=dataset[dataset.columns[1:29]]
point = int(0.6*(len(dataset)))
train = data[0:point+1]
x = StandardScaler().fit_transform(data)
#print(x)


# In[234]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[235]:


from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])


# In[ ]:





# In[236]:


#print(principalDf['principal component 1'][0:858])


# In[237]:


#print(principalDf['principal component 1'].max() + 1)
x_min, x_max = principalDf['principal component 1'].min() - 1, principalDf['principal component 1'].max() + 1
y_min, y_max = principalDf['principal component 2'].min() - 1, principalDf['principal component 2'].max() + 1
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))


# In[238]:


from sklearn import neighbors, datasets
tz = dataset['class']
#Z = tz[0:1526]
#print(tz)
n_neighbors=1
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(principalDf,tz)


# In[239]:


Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
U = labels
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(principalDf['principal component 1'][valend+1:], principalDf['principal component 2'][valend+1:], c=U, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("2-Class classification")
figsize = plt.rcParams["figure.figsize"]
figsize[0]=25
figsize[1]=20
plt.rcParams["figure.figsize"] = figsize
plt.show()
#plt.pcolormesh(xx, yy, Z, cmap=cmap_light)


# In[240]:


Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#U = labels
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], c=tz, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("2-Class classification")
figsize = plt.rcParams["figure.figsize"]
figsize[0]=25
figsize[1]=20
plt.rcParams["figure.figsize"] = figsize
plt.show()
#plt.pcolormesh(xx, yy, Z, cmap=cmap_light)


# In[ ]:





# In[ ]:




