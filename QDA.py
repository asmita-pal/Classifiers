# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 17:58:34 2017

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

def covariance(x):
    
    #Sample means for each class
    ui = np.mean(x, axis = 0)
 
    #Class covariance
    z = 0.0
    for xi in x:
        xi, ui = xi.reshape(2,1), ui.reshape(2,1)
        z = z + (xi-ui).dot((xi-ui).T)
    N_ = len(x)
    Rhat = 1.0 / (N_ - 1) * (z)
    return Rhat
    
def error_calc(X0,X1):
    Rhat0 = covariance(X0)
    Rhat1 = covariance(X1)    
    X = np.concatenate((X0, X1), axis = 0)
    
    u0 = np.mean(X0, axis = 0)
    u1 = np.mean(X1, axis = 0)
    
    delta_U0 = delta(X, u0, Rhat0)
    delta_U1 = delta(X, u1, Rhat1)


    #Calculating total length
    N_total = len(X)
    error = 0.0
    
    for i,j in zip(range(0, len(X0)),range(len(X1)-1, N_total)):
        if (delta_U1[i] > delta_U0[i]):
            error += 1
        if (delta_U0[j] > delta_U1[j]):
            error += 1
    error_rate = error / N_total
    return error_rate

def delta(X, U, Rhat):
    delta_k = []
    for x in X:
        #print x
        part1 = math.log(0.5)
        part2 = math.log(np.linalg.det(Rhat))
        part3 = ((x-U).T).dot(np.linalg.pinv(Rhat)).dot(x-U)
        delta_k.append(part1 - (part2 / 2.0 ) - (part3 / 2.0))
    return np.array(delta_k)

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
plt.title("Quadratic Discriminant Analysis")

###########################################
#Quadratic Discriminant Analysis

#Build X matrix
X0 = x0.T
X1 = x1.T

#Sample means for each class
U0 = np.mean(X0, axis = 0)
U1 = np.mean(X1, axis = 0)

#Class covariance
Rhat0 = covariance(X0)
Rhat1 = covariance(X1)

error_QDA_train = error_calc(X0, X1)
print "QDA Error for training data: ",error_QDA_train

#Generate test data
Ntest0 = 5000
Ntest1 = 5000

xtest0 = gendat2(0, Ntest0)
xtest1 = gendat2(1, Ntest1)
nerr = 0.0

#plt.plot(xtest0[:,0], xtest0[:,1], 'gx')
#plt.plot(xtest1[:,0], xtest1[:,1], 'ro')

#Calculate error for test data

error_QDA_test = error_calc(xtest0, xtest1)
print "QDA Error for test data: ", error_QDA_test

#Plot classification regions
xmin = np.vstack([x0[0], x1[0]]).min()
ymin = np.vstack([x0[1], x1[1]]).min()

xmax = np.vstack([x0[0], x1[0]]).max()
ymax = np.vstack([x0[1], x1[1]]).max()

xpl = np.linspace (xmin,xmax,100)
ypl = np.linspace (ymin,ymax,100)
redpts = []
greenpts = []

Rhat_0 = covariance(X0)
Rhat_1 = covariance(X1)

for x in xpl:
    for y in ypl:
        arr = np.array([[x,y]])
        d0 = delta(arr, U0, Rhat_0)
        d1 = delta(arr, U1, Rhat_1)
        if(d0 > d1):
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