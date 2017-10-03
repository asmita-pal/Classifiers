# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:59:09 2017

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
plt.title("Linear Regression")

###########################################
#Linear regression classifier

#Build X matrix
X = np.concatenate(((np.concatenate((np.ones((N0,1), float), x0.transpose()), axis = 1)), 
                   (np.concatenate((np.ones((N1,1), float), x1.transpose()), axis =1))), axis =0)

#Build Y matrix
Y = np.concatenate(((np.concatenate((np.ones((N0,1), float), np.zeros((N0,1), float)), axis=1)), 
                     (np.concatenate((np.zeros((N1,1), float), np.ones((N1,1), float)), axis =1))), axis = 0)

#Parameter Matrix
Bhat = np.linalg.solve(np.matmul(X.transpose(), X), 
                       np.matmul(X.transpose(), Y))

#Approximate Response
Yhat = np.matmul(X, Bhat)
Yhathard = Yhat > 0.5

nerr = ((abs(Yhathard - Y)).sum()) / 2
errrate_linregress_train = nerr / N
print "Linear Regression Error for training data: ", errrate_linregress_train
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

for i in range(0, Ntest0):
    yhat = np.matmul(xtest0[i].transpose(), Bhat)
    if (yhat[1] > yhat[0]):
        nerr = nerr + 1
        
for i in range(0, Ntest1):
    yhat = np.matmul(xtest1[i].transpose(), Bhat)
    if ( yhat[0] > yhat[1]):
        nerr= nerr + 1

errrate_linregress_test = nerr / (Ntest0 + Ntest1)
print "Linear Regression Error for training data: ", errrate_linregress_test

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
        yhat = np.matmul([1, x, y], Bhat)
        if(yhat[0] > yhat[1]):
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