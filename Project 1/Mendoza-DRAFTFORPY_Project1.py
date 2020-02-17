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

# In[234]:


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


# In[125]:


trX_array = Numpyfile['trX']
trX_array


# In[272]:


trY_array = Numpyfile['trY']
trY_array


# In[280]:


tsX_array = Numpyfile['tsX']
tsX_array


# In[281]:


tsY_array = Numpyfile['tsY']
trY_array


# In[129]:


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

# In[216]:


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


# In[217]:


trX_newArray = np.array(trX_newList)
print("SHAPE OF TOTAL TRAINING DATA, trX_new: " + str(trX_newArray.shape))
trX_newArray


# -------------------------

# In[221]:


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


# In[220]:


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
# ## Calculate Mean, Stdev & Covariance for SEVEN (TRAINING)

# In[223]:


#Convert trX SEVEN into list
trX_newList7 = trX_newList[0:6265] #Split from 6265 rows

trX_sevenNPArray = np.array(trX_newList7)
print("SHAPE OF trX SEVEN SET: " + str(trX_sevenNPArray.shape))
trX_newList7


# In[239]:


#Calculate mean for SEVEN
#Calculate std for SEVEN

trX_tempMeanHolder_SEVEN = []
trX_tempSTDHolder_SEVEN = []
 
trX_tempMeanHolder_SEVEN = [i[0] for i in trX_newList7]
trX_tempSTDHolder_SEVEN = [i[1] for i in trX_newList7]

#(!!!) CALCULATE MEAN OF MEANS and MEAN OF STDEV
trX_mean_SEVEN = statistics.mean(trX_tempMeanHolder_SEVEN)
trX_std_SEVEN = statistics.mean(trX_tempSTDHolder_SEVEN)

#Store mean and standard deviaion for SEVEN as answers in list
trX_calc_SEVEN = [trX_mean_SEVEN, trX_std_SEVEN]

print("trX MEAN OF MEAN 7: " + str(trX_mean_SEVEN))
print("trX MEAN OF STD 7: " + str(trX_std_SEVEN))
trX_calc_SEVEN


# ----------------------------------------------------------------
# ## Calculate Mean, Stdev & Covariance for SEVEN (TESTING)

# In[231]:


#Convert trX SEVEN into list
tsX_newList7 = tsX_newList[0:1028] #Split from 6265 rows

tsX_sevenNPArray = np.array(tsX_newList7)
print("SHAPE OF tsX SEVEN SET: " + str(tsX_sevenNPArray.shape))
tsX_newList7


# In[240]:


#Calculate mean for SEVEN
#Calculate std for SEVEN

tsX_tempMeanHolder_SEVEN = []
tsX_tempSTDHolder_SEVEN = []
 
tsX_tempMeanHolder_SEVEN = [i[0] for i in trX_newList7]
tsX_tempSTDHolder_SEVEN = [i[1] for i in trX_newList7]

#(!!!)CALCULATE MEAN OF MEANS
tsX_mean_SEVEN = statistics.mean(tsX_tempMeanHolder_SEVEN)
tsX_std_SEVEN = statistics.mean(tsX_tempSTDHolder_SEVEN)

#Store mean and standard deviaion for SEVEN as answers in list
tsX_calc_SEVEN = [tsX_mean_SEVEN, tsX_std_SEVEN]

print("tsX MEAN OF MEANS 7: " + str(tsX_mean_SEVEN))
print("tsX MEAN OF STD 7: " + str(tsX_std_SEVEN))
tsX_calc_SEVEN


# ----------------------------------------------------

# In[256]:


#Calculate Covariance Matrix for SEVEN. Done for MEAN OF MEANS and MEAN of STDEV

#General covariance function
def cov_trX_SEVEN(a, b):

    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum/(len(a)-1)

trX_seven_COVARIANCE = cov_trX_SEVEN(trX_tempMeanHolder_SEVEN, trX_tempSTDHolder_SEVEN)

#Pulled the original lists HOLDING the means and the standard deviations
print("COVARIANCE trX SEVEN: " + str(cov_trX_SEVEN(trX_tempMeanHolder_SEVEN, trX_tempSTDHolder_SEVEN)))
    #Proves that the two features are, indeed, independent


# --------------------------------------------------------------------------------------------------
# ## Calculate Mean, Stdev and Covariance for EIGHT (TRAINING)

# In[229]:


#Convert trX EIGHT into list
trX_newList8 = trX_newList[6265:12117] #One more past 12116. Split into 5851 rows

trX_eightNPArray = np.array(trX_newList8)
print("SHAPE OF trX EIGHT SET: " + str(trX_eightNPArray.shape))
trX_newList8


# In[241]:


#Calculate mean for EIGHT
#Calculate std for EIGHT

trX_tempMeanHolder_EIGHT = []
trX_tempSTDHolder_EIGHT = []
 
trX_tempMeanHolder_EIGHT = [j[0] for j in trX_newList8]
trX_tempSTDHolder_EIGHT = [j[1] for j in trX_newList8]

#(!!!) CALCULATE MEAN OF MEANS
trX_mean_EIGHT = statistics.mean(trX_tempMeanHolder_EIGHT)
trX_std_EIGHT = statistics.mean(trX_tempSTDHolder_EIGHT)

#Store mean and standard deviaion for EIGHT as answers in list
trX_calc_EIGHT = [trX_mean_EIGHT, trX_std_EIGHT]

print("trX MEAN OF MEANS 8: " + str(trX_mean_EIGHT))
print("trX MEAN OF STD 8: " + str(trX_std_EIGHT))
trX_calc_EIGHT


# ## Calculate Mean, Stdev & Covariance for EIGHT (TESTING)

# In[232]:


#Convert trX EIGHT into list
tsX_newList8 = tsX_newList[1028:2003] #One more past 12116. Split into 5851 rows

tsX_eightNPArray = np.array(tsX_newList8)
print("SHAPE OF tsX EIGHT SET: " + str(tsX_eightNPArray.shape))
trX_newList8


# In[242]:


#Calculate mean for EIGHT
#Calculate std for EIGHT

tsX_tempMeanHolder_EIGHT = []
tsX_tempSTDHolder_EIGHT = []
 
tsX_tempMeanHolder_EIGHT = [j[0] for j in tsX_newList8]
tsX_tempSTDHolder_EIGHT = [j[1] for j in tsX_newList8]

#(!!!) CALCULATE MEAN OF MEANS
tsX_mean_EIGHT = statistics.mean(tsX_tempMeanHolder_EIGHT)
tsX_std_EIGHT = statistics.mean(tsX_tempSTDHolder_EIGHT)

#Store mean and standard deviaion for EIGHT as answers in list
tsX_calc_EIGHT = [tsX_mean_EIGHT, tsX_std_EIGHT]

print("tsX MEAN OF MEANS 8: " + str(tsX_mean_EIGHT))
print("tsX MEAN OF STD 8: " + str(tsX_std_EIGHT))
tsX_calc_EIGHT


# ------------------------------------------

# In[255]:


#Calculate Covariance Matrix for EIGHT. Done for MEAN OF MEANS and MEAN of STDEV

#General covariance function
def cov_trX_EIGHT(a, b):

    if len(a) != len(b):
        return

    #Calculated the mean of EACH feature
    a_mean = np.mean(a)
    b_mean = np.mean(b)

    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum/(len(a)-1)

trX_eight_COVARIANCE = cov_trX_EIGHT(trX_tempMeanHolder_EIGHT, trX_tempSTDHolder_EIGHT)

#Pulled the original lists HOLDING the means and the standard deviations
print("COVARIANCE trX EIGHT: " + str(cov_trX_EIGHT(trX_tempMeanHolder_EIGHT, trX_tempSTDHolder_EIGHT)))
    #Proves that the two features are, indeed, independent


# In[259]:


#Calculate the accuracy values 

def p_x_given_y(x, mean_y, variance_y):

    # Arguments placed into probability desnity function
    term1 = 1/(np.sqrt(2*np.pi*variance_y))
    term2 = np.exp((-(x-mean_y)**2)/(2*variance_y))
    p = term1 * term2
    
    return p


# -------------------------------------

# In[372]:


#Calculate priors from TRAINING SET
    #TRAINING: 7 is from 0-6265 (6266 total)
    #TRAINING: 8 is from 6266-12116 (5851 total)
    #TRAINING: Total rows of 12116

n_SEVEN = 6266
n_EIGHT = 5851
total_NUM = 12116
    
p_SEVEN = 6266/12116
p_EIGHT = 5851/12116

print("tsX MEAN OF MEANS 7: " + str(tsX_mean_SEVEN))
print("tsX MEAN OF STDev 7: " + str(tsX_std_SEVEN))
print(trX_calc_SEVEN)
print()

print("trX MEAN OF MEANS 8: " + str(trX_mean_EIGHT))
print("trX MEAN OF STDev 8: " + str(trX_std_EIGHT))
print(trX_calc_EIGHT)
print()


# In[351]:


featureVector = trY_array.tolist()
featureVectorALL = featureVector[0]
sevenVector = featureVectorALL[0:6265]
#sevenVector


# In[352]:


featureVector = trY_array.tolist()
featureVector1 = featureVector[0]
eightVector = featureVectorALL[6265:12116]
eightVector


# In[368]:


# Calculate the Gaussian probability distribution function for x

# Example of Gaussian PDF
from math import sqrt
from math import pi
from math import exp

trX_seven_COVARIANCE = cov_trX_SEVEN(trX_tempMeanHolder_SEVEN, trX_tempSTDHolder_SEVEN)
trX_eight_COVARIANCE = cov_trX_EIGHT(trX_tempMeanHolder_EIGHT, trX_tempSTDHolder_EIGHT)

#(!!!)MUST USE LISTS INSTEAD OF NUMPY ARRAYS
    #Need one for 7, need one for 8
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

totalLen = len(actualVector)
print("LENGTH IS: " + str(totalLen))

aCounter = 0
bCounter = 0

#Place in standard deviation to calculate probability
for i in actualVector:
    a = calculate_probability(i, tsX_mean_SEVEN, tsX_std_SEVEN)
    b = calculate_probability(i, tsX_mean_EIGHT, trX_std_EIGHT)

    print("SEVEN is: " + str(a))
    print("EIGHT is: " + str(b))
    
    if a > b:
        print("IT IS 7")
        aCounter += 1
    else:
        print("IT IS 8")
        bCounter += 1


# In[371]:


print("SEVEN PROBABILITY: " + str(aCounter/total_NUM))
print("EIGHT PROBABILITY: " + str(bCounter/total_NUM))

#Hence, because there are greater instances of seven to eight, it belongs at 7


# # PART 3: Logistic Regression using Gradient Ascent
# * Report classification accuracy for "7" and "8"
# * Produce predicted label for each testing sample
# * (!!!) GRADIENT ASCENT MUST BE DONE FROM SCRATCH
# 
# 
# * DELIVER
#     * Predicted Labels
#     * Classification accuracy for "7" and "8"

# In[328]:




np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))

plt.figure(figsize=(12,8))
plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
            c = simulated_labels, alpha = .4)


# In[340]:


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        
    weights = np.zeros(features.shape[1])
    
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with log likelihood gradient
        output_error_signal = target - predictions
        
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 10000 == 0:
            print(log_likelihood(features, target, weights))
        
    return weights

weights = logistic_regression(simulated_separableish_features, simulated_labels,
                     num_steps = 50000, learning_rate = 5e-5, add_intercept=True)

final_scores = np.dot(np.hstack((np.ones((simulated_separableish_features.shape[0], 1)),
                                 simulated_separableish_features)), weights)
preds = np.round(sigmoid(final_scores))

print('Accuracy from scratch: {0}'.format((preds == simulated_labels).sum().astype(float) / len(preds)))


# In[ ]:




