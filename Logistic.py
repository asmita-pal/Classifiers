# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 23:30:12 2017

@author: Asmita
"""

import math
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

import time
start_time = time.time()

#Generate test data function
def gendat2(class_x, N):
    m0   = [ [-0.132, 0.320, 1.672, 2.230, 1.217, -0.819,  
         3.629,  0.8210,  1.808, 0.1700], [-0.711, -1.726, 
         0.139, 1.151, -0.373, -1.573, -0.243, -0.5220, 
         -0.511, 0.5330]]
    m1   =  [[-1.169, 0.813, -0.859, -0.608, -0.832, 2.015, 
         0.173, 1.432,  0.743, 1.0328], [2.065, 2.441,  
         0.247,  1.806,  1.286, 0.928, 1.923, 0.1299, 
         1.847, -0.052]]
    x  =  []
    for i in range (0, N): #   draw N points
        idx = int(random.rand()*10 +  1)
        #print idx
        idx = idx -1
        m0, m1 = np.array(m0), np.array(m1)
        if(class_x == 0):
            m  = m0[:,idx]
        elif(class_x==1):
            m  = m1[:,idx]
        x.append(m+ random.standard_normal((1,))/math.sqrt(5))
    return np.array(x)

#Initialize p(xi;B), i.e., the sigmoid function
def sigmoid(x, B):
    z = (B.T).dot(x)
    return (np.exp(z) / (1.0 + np.exp(z)))

#Calculate the log_likelihood
def log_likelihood(x, y, B):
    l_B = 0.0
    for xi, yi in zip(x, y):
        xi =xi.T #Since according to problem x and Beta is a column vector
        l_B = l_B + yi * ((B.T).dot(xi)) - np.log(1 + np.exp((B.T).dot(xi)))
    return l_B

def gradient(x, y, B):
    d = 0.0
    for xi, yi in zip(x,y):
        probability = sigmoid(xi.T,B)
        d = d + ((yi - probability) * xi.T)
    return d

def hessian(x, y, B):
    #h = np.zeros((3,3), float)
    h = 0.0
    for xi, yi in zip(x, y):
        xi = xi.reshape(3,1)
        probability = sigmoid(xi, B)
        h = h + (probability * (1 - probability)* (xi).dot(xi.T))
    return (-h)

#np.seterr(all = 'ignore')

#Load data
data = np.loadtxt('classasgntrain1.dat')
x0 = (np.array(data[:,[0,1]])).transpose()
N0 = x0.shape[1]
x1 = (np.array(data[:,[2,3]])).transpose()
N1 = x1.shape[1]
N = N0 + N1

#plot training data
plt.figure(figsize=(12,9))
plt.plot(x0[0,], x0[1,], 'gx', markersize = 10)
plt.plot(x1[0,], x1[1,], 'ro', markersize = 10)
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.title("Linear Logistic Regression")

###########################################
#Linear logistic Regression classifier

#Build X matrix
X0 = np.concatenate((np.ones((N0,1), float), x0.transpose()), axis = 1)
X1 = np.concatenate((np.ones((N1,1), float), x1.transpose()), axis =1)
X = np.concatenate((X0, X1), axis = 0)

#Build Y matrix
Y = np.concatenate((np.ones((N0,1), float), np.zeros((N1,1), float)), axis=0)

#Initialize Beta
Beta = np.array([0.0,0.0, 0.0])

#Update Beta and log likelihood function
iteration =0
B_old = Beta
del_l = np.Infinity
l_b = log_likelihood(X, Y, Beta)
while abs(del_l) > .0000000001 and iteration < 20:
    iteration += 1
    derivative = gradient(X, Y, Beta)
    H = hessian(X,Y, Beta)
    H_inverse = np.linalg.inv(H)
    Beta = Beta - H_inverse.dot(derivative)
    l_b_new = log_likelihood(X, Y, Beta)
    del_l = l_b - l_b_new
    l_b = l_b_new

error = 0.0
i=0

#Calculate error for traning data

for xi, i in zip(X, range(0,N)):
    probability_0 = sigmoid(xi.T, Beta)
    probability_1 = 1 - probability_0
    #print xi, probability_0, probability_1
    if i < N0 and probability_1 > probability_0:
        error+= 1
    elif (i > N0 and probability_0 > probability_1):
        error+=1

error_logistic_regression = error / N
print "Logistic Regression error for training data: ",error_logistic_regression

#Generate test data
Ntest0 = 5000
Ntest1 = 5000

xtest0 = gendat2(0, Ntest0)
xtest1 = gendat2(1, Ntest1)
nerr = 0.0

#plt.plot(xtest0[:,0], xtest0[:,1], 'gx')
#plt.plot(xtest1[:,0], xtest1[:,1], 'ro')

#Form new array by concatenating ones to test data for class 1 and class 0
xtest0 = (np.concatenate((np.ones((Ntest0,1), float), xtest0), axis = 1))
xtest1 = (np.concatenate((np.ones((Ntest1,1), float), xtest1), axis = 1))

#Calculate error for test data

error = 0.0
xtest = np.concatenate((xtest0, xtest1), axis = 0)
for xi, i in zip(xtest, range(0, len(xtest))):
    probability_0 = sigmoid(xi.T, Beta)
    probability_1 = 1 - probability_0
    #print xi, probability_0, probability_1
    if i < len(xtest0) and probability_1 > probability_0:
        error+= 1
    elif (i > len(xtest0) and probability_0 > probability_1):
        error+=1

error_logistic_regression_testdata = error / (Ntest0 + Ntest1)
print "Logistic Regression Error for test data: ",error_logistic_regression_testdata

#Plot classification regions
xmin = np.vstack([x0[0], x1[0]]).min()
ymin = np.vstack([x0[1], x1[1]]).min()

xmax = np.vstack([x0[0], x1[0]]).max()
ymax = np.vstack([x0[1], x1[1]]).max()

xpl = np.linspace (xmin,xmax,100)
ypl = np.linspace (ymin,ymax,100)
redpts = []
greenpts = []

for x in xpl:
    for y in ypl:
        arr = np.array([[1], [x] ,[y]])
        probability_0 = sigmoid(arr, Beta)
        probability_1 = 1 - probability_0
        if probability_0 > probability_1:
            greenpts.append([x,y])
        else:
            redpts.append([x,y])

greenpts = np.array(greenpts)
redpts = np.array(redpts)
plt.plot(greenpts[:,0], greenpts[:,1] , 'g.' , markersize = 3.5)
plt.plot(redpts[:,0], redpts[:,1], 'r.', markersize = 3.5)
plt.axis('tight')
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))