# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:15:23 2017

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
        idx = idx -1
        m0, m1 = np.array(m0), np.array(m1)
        if(class_x == 0):
            m  = m0[:,idx]
        elif(class_x==1):
            m  = m1[:,idx]
        x.append(m+ random.standard_normal((1,))/math.sqrt(5))
    return np.array(x)

def find_neighbor(k, x, X, y):
    distance = 0.0
    arr = []
    Y_ = y 
    kneighbor = []
    for xi, yi in zip(X, Y_):
        distance = math.sqrt((x[0] - xi[0])**2 + (x[1]- xi[1])**2)
        arr.append([xi[0], xi[1], yi[0], distance])
    arr = sorted(arr, key=lambda distance:distance[-1])
    for i in range(0,k):
        kneighbor.append([arr[i][0], arr[i][1], arr[i][2]])
    return kneighbor

def error_calc(k, X_data, y):
    error = 0.0
    i = 0
    l = len(X_data)
    for xi, yi in zip(X_data,y):
        i+=1
        knearest=np.array(find_neighbor(k, xi, X,y))
        fx = (np.sum(knearest, axis = 0))[2] / k
        if(fx > 0.5):
            x_class = 1
        else:
            x_class = 0
        if x_class != yi[0]:
            error += 1
    error_rate = error / l
    return error_rate

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
plt.title("k-Nearest Neighbor Classifier")

#Build X matrix
X0 = x0.T
X1 = x1.T
X = np.concatenate((X0, X1), axis = 0)

#Build Y matrix
Y = np.concatenate((np.ones((N0,1), float), np.zeros((N1,1), float)), axis=0)

k = 1
error_knearest_train = error_calc(k, X,Y)
print "k-nearest Neighbor Error for training data: ",error_knearest_train

#Generate test data
Ntest0 = 5000
Ntest1 = 5000

xtest0 = gendat2(0, Ntest0)
xtest1 = gendat2(1, Ntest1)

#plt.plot(xtest0[:,0], xtest0[:,1], 'gx')
#plt.plot(xtest1[:,0], xtest1[:,1], 'ro')

#Calculate error for test data
xtest = np.concatenate((xtest0,xtest1), axis = 0)
ytest = np.concatenate((np.ones((Ntest0,1), float), np.zeros((Ntest1,1), float)), axis=0)

error_test = error_calc(k, xtest, ytest)
print "k-nearest Neighbor Error for test data: ", error_test

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
        arr = np.array([x,y])
        knearest=np.array(find_neighbor(k, arr, X,Y))
        fx = (np.sum(knearest, axis = 0))[2] / k
        if(fx > 0.5):
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