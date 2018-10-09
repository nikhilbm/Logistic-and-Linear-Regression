from __future__ import division
from  numpy.linalg import inv

import numpy as np
import sys
import pylab
import matplotlib.pyplot as plt
import math



def f(x1,theta):
	return(theta[0]+theta[1]*x1)


def linreg(x_design,y_train):
	part1 = np.matmul(x_design.transpose(),x_design)
	part1inv = inv(part1)
	res  = np.matmul(part1inv, x_design.transpose())
	newy = y_train.reshape((450,1))
	return (np.matmul(res,newy))


def constructdiag(x,i,tau = 5):
	W = np.diag(np.exp(-((x - x[i])**2)/(2*tau*tau)))
	return W

def wlinreg(x_design, y_train):
	preds = []
	for i in range(0,450):
		W = constructdiag(x_design[:,1],i)
		XTW = np.matmul(x_design.transpose(),W)
		prod1 = inv(np.matmul(XTW,x_design))
		newy = y_train.reshape((450,1))
		prod2 = np.matmul(XTW,newy)
		theta = np.matmul(prod1,prod2)
		val = np.dot(theta[:,0], x_design[i,:])
		preds.append(val)
	return preds



if __name__ == '__main__':
	#Load the full matrix
	fullmat  =  np.genfromtxt('quasar_train.csv',delimiter = ",")
	#Extract first 2 rows
	datamtx = fullmat[0:2]
	#print datamtx.shape
	x_train = datamtx[0,:]
	y_train = datamtx[1,:]
	#print x_train.shape
	ones = np.ones((450,1),dtype = 'float64')
	x_design = np.append(ones,x_train.reshape((450,1)),axis = 1) #450x2 matrix
	#for k, x_eval in enumerate(x_design):
	#	print x_eval 
	#print x_design.shape
	#print x_design[0][0]
	#plotting the data points
	plt.plot(x_train, y_train,'ro')
	plt.xlabel('Wavelength in Angstrom')
	plt.ylabel('Luminous flux')
	#plt.show ()
	#print np.max(y_train)
	#print  np.min(y_train)
	ypred = wlinreg(x_design,y_train)
	theta = linreg(x_design, y_train)
	#print theta
	x1 = np.arange(1150,1600,1)
	plt.plot(x1,ypred,linewidth = 3)
	#plt.plot(x1, f(x1,theta))
	plt.show()
