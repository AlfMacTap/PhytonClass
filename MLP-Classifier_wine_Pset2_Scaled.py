#!/usr/bin/env python
# coding: utf-8

# In[182]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[183]:


# import the data set
from sklearn.datasets import load_wine
wine=load_wine()


# In[184]:


# feature matrix
X=wine.data
y=wine.target
y


# In[185]:


X


# In[186]:


wine.feature_names


# In[187]:


# corroborate the shape of the arrays
print(np.shape(X))
print(np.shape(y))


# # Train and test data

# In[188]:


# split the data into training and test sets
from sklearn.model_selection import train_test_split


# In[189]:


# following code is to create the training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)
#'test_size': If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.
#'random_state': If int, random_state is the seed used by the random number generator;


# # Scaling

# Before making actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated. Feature scaling is performed only on the training data and not on test data. This is because in real world, data is not scaled and the ultimate purpose of the neural network is to make predictions on real world data. Therefore, we try to keep our test data as real as possible.

# In[190]:


# scale the training data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# # Multo-layer Perceptron (MLP) - Classifier

# 1. Following the method that we have discussed in class, build an MLPClassifier model to predict the target variable from the feature data.

# In[191]:


# now build the neural network using MLPClassifier
from sklearn.neural_network import MLPClassifier


# In[204]:


# The following code will create a classifier with three layers of 10 nodes each 
# 'max_iter' refers to the number of epochs, one epoch is a combination of one cycle of feed-forward and back propagation phase.
mlp = MLPClassifier(hidden_layer_sizes=(3,3,3), max_iter=25000)


# In[205]:


# now fit the classifier to the training data
mlp.fit(X_train,y_train)


# In[206]:


# makign some predictions using your MLPClassifier
predictions=mlp.predict(X_test)


# In[207]:


# test predictions
from sklearn.metrics import classification_report, confusion_matrix


# In[208]:


print(confusion_matrix(y_test,predictions))


# In[209]:


print(classification_report(y_test,predictions))


# # Build a loop to test how well different number of hidden layers performs

# 2.1. You will explore the parameter space of the MLPClassifier model and determine the combination(s) of number of layers and number of neurons that yield the best model accuracy score.

# In[210]:


from sklearn import metrics


# In[211]:


scores={}
scores_list=[]
l_range=range(4,20,1)
for l in l_range:
    print(l)
    mlp=MLPClassifier(hidden_layer_sizes=(l,l,l),max_iter=2500)
    mlp.fit(X_train,y_train)
    y_pred=mlp.predict(X_test)
    scores[1]=metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))


# 2.2. Produce a figure that illustrates the results of your analysis.

# In[212]:


plt.plot(scores_list)
plt.ylabel('accuracy score')


# # Build a loop to test how well different number of hidden layers performs on a model with 4 and 5 layers

# In[213]:


scores={}
scores_list=[]
l_range=range(4,20,1)


# In[214]:


for l in l_range:
    mlp=MLPClassifier(hidden_layer_sizes=(l,l,l,l),max_iter=2000)
    mlp.fit(X_train,y_train)
    y_pred=mlp.predict(X_test)
    scores[1]=metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))

plt.plot(scores_list)
plt.ylabel('accuracy score')


# In[215]:


scores={}
scores_list=[]
l_range=range(4,20,1)


# In[216]:


for l in l_range:
    mlp=MLPClassifier(hidden_layer_sizes=(l,l,l,l,l),max_iter=2000)
    mlp.fit(X_train,y_train)
    y_pred=mlp.predict(X_test)
    scores[1]=metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))

plt.plot(scores_list)
plt.ylabel('accuracy score')


# 2.3 Write a short paragraph explaining your results. If more than one combination of layers/neurons results in a high accuracy score (>0.9), discuss which combination of layers/neurons you would use and why.

# 
#     According to the accuracy score from the model with three hidden layers and different number of 'nodes' (figure above), the models with 4,8,11 and 14 nodes have the highest accuracy score. 
# 
#     The accuracy score was similar when using four and five hidden layers and more than two nodes (two last figures).
# 
# 

# # Test your predictions

# 3.1 For each combination of layers/neurons that yields an accuracy score > 0.9, calculate the confusion matrix, precision, recall and f1-score for each target class.

# In[218]:


mlp = MLPClassifier(hidden_layer_sizes=(4,4,4), max_iter=2000)
mlp.fit(X_train, y_train)
predictions=mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[221]:


mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), max_iter=2000)
mlp.fit(X_train, y_train)
predictions=mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[222]:


mlp = MLPClassifier(hidden_layer_sizes=(11,11,11), max_iter=2000)
mlp.fit(X_train, y_train)
predictions=mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[223]:


mlp = MLPClassifier(hidden_layer_sizes=(14,14,14), max_iter=2000)
mlp.fit(X_train, y_train)
predictions=mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# 3.2 Describe the
# implications of your results for model performance for each target class and choose
# which layers/neurons you will use in the final version of your model.

#     Results from models of three layers and 4, 8, 11 or 12 nodes show similar results in terms of precision, recall, and f1-score. However, among them, the model with 8 nodes has a ‘perfect’ performance. 
#     Based on the above results, I will use a model with three hidden layers and 8 nodes for further steps on this problem set.
# 

# # Final model

# 4. Use the final version of your model, developed using the steps above, to predict the
# class for the ‘unknown’ samples found in the ‘unknow_wine.csv’ file.

# In[231]:


mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), max_iter=2000)
mlp.fit(X_train, y_train)
predictions=mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[232]:


# import 'unknown' samples
infile='C:/Users/a_mac/Desktop/unknown_wine.csv'
data=pd.read_csv(infile)


# In[236]:


classes = {0:'wine1',1:'wine2',2:'wine3'}


# In[245]:


y_predict=mlp.predict(data)


# In[246]:


print(classes[y_predict[0]])
print(classes[y_predict[1]])
print(classes[y_predict[2]])
print(classes[y_predict[3]])
print(classes[y_predict[5]])
print(classes[y_predict[6]])
print(classes[y_predict[7]])
print(classes[y_predict[8]])
print(classes[y_predict[9]])


# # Probability

# 5.1 Find a scikit-learn function that will give the probability that each unknown wine fits into each of one of the three target classes.

# In[243]:


mlp.predict_proba(data)
#use 'predict_proba(self,X)'
#Parameters: X:{array-like, sparse matrix} of shape (n_samples, n_features) The input data.
#Returns: y_prob: ndarray of shape (n_samples, n_classes) The predicted probability of the sample for each class in the model, where classes are ordered as they are in self.classes_.        


# 5.2 List the probabilities for each unknown sample for each target class and comment on your results.

#     According to the model, all but one of the unknowns correspond to the one type of wine (‘wine1’). The results from the ‘confusion matrix’, ‘classification report’, and probability test, suggest that predictions done by the model should be correct. However, other steps can be applied (i.e. building models holding different sets of training and testing data) for further corroboration of the model.

# In[ ]:




