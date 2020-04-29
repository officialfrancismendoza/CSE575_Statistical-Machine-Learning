#!/usr/bin/env python
# coding: utf-8

# **PROGRAMMER:** Francis Mendoza
# 
# **ASUID:** 1213055998
# 
# **EMAIL:** fmendoz7@asu.edu
# 
# **NOTE:** This Jupyter Notebook file is provided **FOR REFERENCE**, as all the code was developed & successfully compiled here before conversion into a .py file
# 
# -------------------------------

# # Parameter Estimation For MNIST Dataset
# 
# * Classification for digits "7" and "8" only 
# * 70,000 images of handwritten digits 
#     * 60,000 training images 
#     * 10,000 testing images  
#     
#     
# * Number of samples for TRAINING SET:
#     * "7": 6265
#     * "8": 5851
#     
#     
# * Number of samples in TESTING SET:
#     * "7": 1028
#     * "8": 974
#     
#     
# * Features
#     * Average of ALL pixel values in image
#     * Standard Deviation of all pixel values in image

# In[60]:


#Extract Data Set

from __future__ import print_function
import statistics 

import os
import scipy.io
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Import the relevant data using the filepath
Numpyfile= scipy.io.loadmat('mnist_data.mat') 
    #Dictionary 
    
#Display the NumpyFile to check four matrices 
Numpyfile


# In[61]:


trX_array = Numpyfile['trX']
trX_array


# In[62]:


trY_array = Numpyfile['trY']
trY_array


# In[63]:


tsX_array = Numpyfile['tsX']
tsX_array


# In[64]:


tsY_array = Numpyfile['tsY']
trY_array


# In[65]:


#Checking length
    #6265 + 5851 = 12116
    #1028 + 974 = 2002

#REPRESENTS POINTS
print("trX dimensions: "+ str(Numpyfile['trX'].shape))
print("tsX dimensions: "+ str(Numpyfile['tsX'].shape))

#LABELS FOR RESPECTIVE SETS
print("trY dimensions: "+ str(Numpyfile['trY'].shape))
print("tsY dimensions: "+ str(Numpyfile['tsY'].shape))


# # PART 1: Extract Features & Estimate Parameters
# * Estimate parameters for 2D normal distribution for EACH DIGIT using training data
# * Two distributions- one for each digit 
# * MLE Density Estimation
# 
# 
# * Features
#     * Average of ALL pixel values in image
#     * Standard Deviation of all pixel values in image
#     
#     
# * DELIVER
#     * Extract features for both TRAINING and TESTING set (7 and 8 within one training/testing matrix)
#     * Parameter estimation

# In[66]:


#Feature Selection
"""
REMEMBER: Feature selection is extremely importantby ensuring you are paying attention to
the right features 
    
Paying attention to irrelevant or partially relevant features can negatively impact model performance
"""    
trX_newList = []

#Nested loop to iterate through numpy array of lists
for i in Numpyfile['trX']:
    mean_temp = statistics.mean(i)
    std_temp = statistics.stdev(i)
    #trX_new += [[mean_temp, std_temp]]
    trX_newList += [[mean_temp, std_temp]]
    

"""
#----------------------------------------------------
trX_preList7_PT1 = trX_newList[0:6265]
trXMeanHolder_SEVEN_PT1 = [i[0] for i in trX_preList7_PT1]
trXSTDHolder_SEVEN_PT1 = [i[1] for i in trX_preList7_PT1]

#----------------------------------------------------
trX_preList8_PT1 = trX_newList[6265:12117]
trXMeanHolder_EIGHT_PT1 = [i[0] for i in trX_preList8_PT1]
trXSTDHolder_EIGHT_PT1 = [i[1] for i in trX_preList8_PT1]
#----------------------------------------------------
"""
    
print("List of lists for total TRAINING data, trX_new")
trX_newList


# In[67]:


trX_newArray = np.array(trX_newList)
print("SHAPE OF TOTAL TRAINING DATA, trX_new: " + str(trX_newArray.shape))
trX_newArray


# -------------------------

# In[68]:


tsX_newList = []

for i in Numpyfile['tsX']:
    mean_temp = statistics.mean(i)
    std_temp = statistics.stdev(i)
    #tsX_new += [[mean_temp, std_temp]]
    tsX_newList += [[mean_temp, std_temp]]
    
"""
#----------------------------------------------------
tsX_preList7_PT1 = tsX_newList[0:1028]
tsXMeanHolder_SEVEN_PT1 = [i[0] for i in tsX_preList7_PT1]
tsXSTDHolder_SEVEN_PT1 = [i[1] for i in tsX_preList7_PT1]

#----------------------------------------------------
tsX_preList8_PT1 = tsX_newList[1028:2003]
tsXMeanHolder_EIGHT_PT1 = [i[0] for i in tsX_preList8_PT1]
tsXSTDHolder_EIGHT_PT1 = [i[1] for i in tsX_preList8_PT1]
#----------------------------------------------------
"""

print("List of lists for total TESTING data, tsX_new")
tsX_newList


# In[69]:


tsX_newArray = np.array(tsX_newList)
print("SHAPE OF TOTAL TESTING DATA, tsX_new: " + str(tsX_newArray.shape))
tsX_newArray


# # ----------------------------------------------------------------------------------------

# # PART 2: Naive Bayes using Estimated Distributions
# * Report classification accuracy for "7" and "8"
# * Produce predicted label for each testing sample
# 
# 
# * DELIVER
#     * Predicted Labels
#     * Classification accuracy for "7" and "8"

# ## (!!!) CHECK IF DATA WAS SPLIT CORRECTLY!
# ----------------
# ## >>> Calculate Mean, Stdev for SEVEN (TRAINING)

# In[70]:


temp = trY_array.tolist()
trY_list = temp[0]

trY_list


# In[71]:


#Actual parse the CORRECT values 
# trX_newList = Numpyfile['trX']
# trX_newList7 = []
# trX_newList8 = [] 

# for i in Numpyfile['trX']:
#     if trX_newList[i] == 0.0:
#         trX_newList7.append(trX_newList[i])
#     else:
#         trX_newList8.append(trX_newList[i])
        
# trX_newList7


# In[72]:


#Convert trX SEVEN into list
trX_newList7 = trX_newList[0:6265] #Split from 6265 rows

trX_sevenNPArray = np.array(trX_newList7)
print("SHAPE OF trX SEVEN SET: " + str(trX_sevenNPArray.shape))
trX_newList7


# In[73]:


#Calculate mean for SEVEN
#Calculate std for SEVEN

trX_tempMeanHolder_SEVEN = []
trX_tempSTDHolder_SEVEN = []
 
trX_tempMeanHolder_SEVEN = [i[0] for i in trX_newList7]
trX_tempSTDHolder_SEVEN = [i[1] for i in trX_newList7]

#(!!!) CALCULATE MEAN OF X1 and STDEV of X1
trX_MM_SEVEN = statistics.mean(trX_tempMeanHolder_SEVEN)
trX_SM_SEVEN = statistics.stdev(trX_tempMeanHolder_SEVEN)

#(!!!)CALCULATE MEAN OF X2 and STDEV of X2
trX_MS_SEVEN = statistics.mean(trX_tempSTDHolder_SEVEN)
trX_SS_SEVEN = statistics.stdev(trX_tempSTDHolder_SEVEN)


print("trX MEAN OF MEAN 7: " + str(trX_MM_SEVEN))
print("trX MEAN OF STD 7: " + str(trX_SS_SEVEN))
print()
print("trX STD of MEAN 7: " + str(trX_SM_SEVEN))
print("trX STD of STD 7: " + str(trX_SS_SEVEN))


# --------------------------------------------------------------------------------------------------
# ## Calculate Mean, Stdev and Covariance for EIGHT (TRAINING)

# In[74]:


#Convert trX EIGHT into list
trX_newList8 = trX_newList[6265:12117] #One more past 12116. Split into 5851 rows

trX_eightNPArray = np.array(trX_newList8)
print("SHAPE OF trX EIGHT SET: " + str(trX_eightNPArray.shape))
trX_newList8


# In[75]:


#Calculate mean for EIGHT
#Calculate std for EIGHT

trX_tempMeanHolder_EIGHT = []
trX_tempSTDHolder_EIGHT = []
 
trX_tempMeanHolder_EIGHT = [j[0] for j in trX_newList8]
trX_tempSTDHolder_EIGHT = [j[1] for j in trX_newList8]

#(!!!) CALCULATE MEAN OF MEANS, MEAN OF STD
trX_MM_EIGHT = statistics.mean(trX_tempMeanHolder_EIGHT)
trX_MS_EIGHT = statistics.mean(trX_tempSTDHolder_EIGHT)

#(!!!) CALCULATE STD OF MEANS, STD OF STD
trX_SM_EIGHT = statistics.stdev(trX_tempMeanHolder_EIGHT)
trX_SS_EIGHT = statistics.stdev(trX_tempSTDHolder_EIGHT)

print("trX MEAN OF MEANS 8: " + str(trX_MM_EIGHT))
print("trX MEAN OF STD 8: " + str(trX_MS_EIGHT))
print()
print("trX STD OF MEANS 8: " + str(trX_SM_EIGHT))
print("trX STD OF STD 8: " + str(trX_SS_EIGHT))


# In[76]:


len(trY_list)
trY_list


# In[77]:


tsX_array.shape


# In[78]:


# #You HAVE TO iterate through the tsX matrix. That is the GROUND TRUTH!!
# flat_tsX_list = []
# tsX_LL = tsX_array.tolist()
# tsX_list = temp2[0]

# len(tsX_list)
#---------------------
tsX_LL = tsX_array.tolist()
flat_tsX_list = []

for sublist in tsX_LL:
    for item in sublist:
        flat_tsX_list.append(item)
        
#flat_tsX_list


# In[79]:


tsX_newList = []

#Iterate to get mean and standard deviation values of TEST set
for i in Numpyfile['tsX']:
    mean_temp = statistics.mean(i)
    std_temp = statistics.stdev(i)
    #tsX_new += [[mean_temp, std_temp]]
    tsX_newList += [[mean_temp, std_temp]]
    
tsX_newList


# In[80]:


tsX_FinalMeanVector = [i[0] for i in tsX_newList]
tsX_FinalSTDVector = [i[1] for i in tsX_newList]

len(tsX_FinalMeanVector)


# In[81]:


# #----------------------------------------------------
# tsX_preList7_PT1 = tsX_newList[0:1028]
# tsXMeanHolder_SEVEN_PT1 = [i[0] for i in tsX_preList7_PT1]
# tsXSTDHolder_SEVEN_PT1 = [i[1] for i in tsX_preList7_PT1]

# #----------------------------------------------------
# tsX_preList8_PT1 = tsX_newList[1028:2003]
# tsXMeanHolder_EIGHT_PT1 = [i[0] for i in tsX_preList8_PT1]
# tsXSTDHolder_EIGHT_PT1 = [i[1] for i in tsX_preList8_PT1]
# #----------------------------------------------------

# print(len(tsXMeanHolder_SEVEN_PT1))


# In[82]:


#(!!!)MUST USE LISTS INSTEAD OF NUMPY ARRAYS
    #Need one for 7, need one for 8

from math import sqrt
from math import pi
from math import exp 

def Gaussian(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


# In[83]:


#Iterate through training label set for Gaussian 
sevenCounter = 0
eightCounter = 0

for i in range(len(trY_list)):
    if trY_list[i] == 0.0:
        sevenCounter += 1
    elif trY_list[i] == 1.0:
        eightCounter += 1
#     else:
#         eightCounter += 1

sevenPrior = sevenCounter/len(trY_list)
eightPrior = eightCounter/len(trY_list)

score = []
counterSeven = 0
counterEight = 0

#Mean of means = mean of <FEATURE>
     #Length of FinalMeanVector and FinalSTDVector is the same
for i in range(len(tsX_FinalMeanVector)):
    sevenMeanGaussian = Gaussian(tsX_FinalMeanVector[i], trX_MM_SEVEN, trX_SM_SEVEN)
    sevenSTDGaussian = Gaussian(tsX_FinalSTDVector[i], trX_MS_SEVEN, trX_SS_SEVEN)
    
    eightMeanGaussian = Gaussian(tsX_FinalMeanVector[i], trX_MM_EIGHT, trX_SM_EIGHT)
    eightSTDGaussian = Gaussian(tsX_FinalSTDVector[i], trX_MS_EIGHT, trX_SS_EIGHT)
    
    #Calculate unnormalized probability values, see which one is higher
    sevenProbability = sevenPrior * sevenMeanGaussian * sevenSTDGaussian
    eightProbability = eightPrior * eightMeanGaussian * eightSTDGaussian
    
    #Add proper value to score vector
    if sevenProbability > eightProbability:
        score.append(0.0)
        print("Class 7")
        counterSeven += 1
    elif sevenProbability < eightProbability:
        score.append(1.0)
        print("Class 8")
        counterEight +=1
#     else:
#         score.append(1.0)


# In[84]:


#Mean Gaussian & STD Gaussian for EACH DIGIT
print("Mean Of Seven Gaussian: ", sevenMeanGaussian)
print("STD Of Seven Gaussian: ", sevenSTDGaussian)
print()
print("Mean Of Eight Gaussian: ", eightMeanGaussian)
print("STD Of Eight Gaussian: ", eightSTDGaussian)
print()
#Priors are right 
print("Prior Of Seven Gaussian: ", sevenPrior)
print("Prior Of Eight Gaussian: ", eightPrior)

if counterSeven > counterEight:
    print("More samples lean towards Seven than Eight")
elif counterSeven < counterEight:
    print("More samples lean towards Eight than Seven")


# In[85]:


#Accuracy function

temp3 = tsY_array.tolist()
tsY_list = temp3[0]

accuracyCounter = 0

for i in range(len(tsY_list)):
    if tsY_list[i] == score[i]:
        accuracyCounter += 1

accuracyValue = accuracyCounter/len(tsY_list)
print("Accuracy Is: ", accuracyValue)


# # PART 3: Logistic Regression using Gradient Ascent
# * Report classification accuracy for "7" and "8"
# * Produce predicted label for each testing sample
# * (!!!) GRADIENT ASCENT MUST BE DONE FROM SCRATCH
# 
# 
# * DELIVER
#     * Predicted Labels
#     * Classification accuracy for "7" and "8"

# In[86]:


# #Represent features (mean and standard deviation in this case) to calculate
# trX_meanTotal = [i[0] for i in trX_newList]
# trX_stdTotal = [i[1] for i in trX_newList]

# pt3_meanArray = np.array(trX_meanTotal)
# pt3_stdArray = np.array(trX_stdTotal)

# #Successfully inserted features
# x1 = pt3_meanArray
# x2 = pt3_stdArray

# featureInput = np.vstack((x1, x2)).astype(np.float32)
# labelInput = np.hstack((featureVector))


# In[87]:


# def sigmoidCalc(scores):
#     sigmoidVal = 1 / (1 + np.exp(-scores))
#     return sigmoidVal

# #Calculate likelihood
# def likelihoodCalc(features, target, weights):
#     scores = np.dot(features, weights)
#     logLikelihoodCalc = np.sum( target*scores - np.log(1 + np.exp(scores)) )
#     return logLikelihoodCalc

# def calculateAssorted(paramFeature, targetVal, stepNum, weightVal):
#     for iter in range(stepNum):
#         scoreVal = np.dot(paramFeature, weightVal)
#         predictVal = sigmoidCalc(scoreVal)

#         # Weights updated on rolling basis
#         errorDetection = targetVal - predictVal

#         gradientVal = np.dot(paramFeature.T, errorDetection)
#         weightVal = (lr * gradientVal) + weight

#         # Print likelihood of occurence
#         if iter % 10000 == 0:
#             print(likelihoodCalc(paramFeature, targetVal, weightVal))

# #Log regression with gradient calculation 
# def log_regressionCalc(paramFeature, targetVal, stepNum, lr, interceptOption = False):
#     if interceptOption:
#         intercept = np.ones((paramFeature.shape[0], 1))
#         paramFeature = np.hstack((intercept, paramFeature))
        
#     weightVal = np.zeros(paramFeature.shape[1])
    
#     calculateAssorted(paramFeature, targetVal, stepNum, weightVal)
        
#     return weightVal

# weights = log_regressionCalc(featureInput, labelInput, stepNum = 50000, lr = 5e-5, interceptOption=True)

# param1 = (np.ones((featureInput.shape[0], 1)), featureInput)
# finalValues = np.dot(np.hstack(param1), weights)
# predictionVal = np.round(sigmoidCalc(finalValues))

# predictionCount = len(predictionVal)
# calculatedAccuracyVal = (predictionVal == labelInput).sum().astype(float) / predictionCount
# print('ACCURACY VALUES: {0}'.format(calculatedAccuracyVal))


# ## (!!!) Pure Logistic Regression [Gradient DESCENT, NOT ascent]

# In[151]:


#Conversion either into CSV or adapt into dictionary
from random import seed
from random import randrange
from csv import reader
from math import exp

import pandas as pd 


# In[162]:


#Feature Vectors
X_temp = [i[0] for i in trX_newList]
y_temp = [i[1] for i in trX_newList]

X = np.array(X_temp)
y = np.array(y_temp)

print("X shape: ", X.shape)
print("Y shape: ", y.shape)


# In[164]:


# def sigmoid:
#     sigmoid_result = 1 / (1 + np.exp(-z))
#     return sigmoid_result

    
# def probability_train(self, X):
#     if self.fit_intercept:
#         X = self.__add_intercept(X)


# In[ ]:





# In[ ]:




