# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 00:01:26 2017

@author: Asmita
"""
import time
start_time = time.time()

import math
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

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
        idx = idx -1
        m0, m1 = np.array(m0), np.array(m1)
        if(class_x == 0):
            m  = m0[:,idx]
        elif(class_x==1):
            m  = m1[:,idx]
        x.append(m+ random.standard_normal((1,))/math.sqrt(5))
    return np.array(x)

def parzen_estimate(x, X):
    s = 0
    lmda = 0.5
    for xi  in X:
        if (x-xi) < lmda:
            K = 1 / math.sqrt(2 * np.pi * lmda) * np.exp(-(x - xi)**2 / (2*lmda))
            s = s + K
    fhat_x = s/ (N * lmda)
    return fhat_x

def error_calc(X_0, X_1):
    error = 0.0
    i = 0
    X_ = np.concatenate((X_0, X_1), axis = 0)
    for x in X_:
        fl1=(parzen_estimate(x[0], X_0[:,0]))
        fl2=(parzen_estimate(x[1], X_0[:,1]))
        fj1=(parzen_estimate(x[0], X_1[:,0]))
        fj2=(parzen_estimate(x[1], X_1[:,1]))
        p_0 = fl1 * fl2
        p_1 = fj1 * fj2
        #Prior probabilities are same
        if p_1 > p_0 and i< len(X_0):
            error += 1
        if p_0 > p_1 and i > len(X_0):
            error += 1
        i += 1
    return error

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
plt.title("Bayes Classifier")

#Build X matrix
X0 = x0.T
X1 = x1.T
X = np.concatenate((X0, X1), axis = 0)

#Calculate training error

error_bayes_train = error_calc(X0, X1) / N
print "Naive Bayes Error for training data: ",error_bayes_train

#Generate test data
Ntest0 = 5000
Ntest1 = 5000

xtest0 = gendat2(0, Ntest0)
xtest1 = gendat2(1, Ntest1)
nerr = 0.0

#plt.plot(xtest0[:,0], xtest0[:,1], 'gx')
#plt.plot(xtest1[:,0], xtest1[:,1], 'ro')

#Calculate error for test data

error_bayes_test = error_calc(xtest0, xtest1) / (Ntest0 +Ntest1)
print "Naive Bayes Error for test data: ",error_bayes_test

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
        fl1=(parzen_estimate(x, X0[:,0]))
        fl2=(parzen_estimate(y, X0[:,1]))
        fj1=(parzen_estimate(x, X1[:,0]))
        fj2=(parzen_estimate(y, X1[:,1]))
        p_0 = fl1 * fl2
        p_1 = fj1 * fj2
        if p_0 > p_1:
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