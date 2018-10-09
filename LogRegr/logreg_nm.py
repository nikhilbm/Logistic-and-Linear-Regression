from __future__ import division

import numpy as np
import sys
import pylab
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv


def g(z):
	val = 1.0/(1+math.exp(-z))
	return val


def h_thetai(x,theta):
	val = np.inner(np.transpose(x),np.transpose(theta))
	# works when both are ordinary row vectors like (i,0)
	return g(val)

def costfn(x,y,theta):
	val = 0
	for k in range(99):
		val += math.log(1.0/h_thetai(y[k][0]*x[k,:],theta))
	avg = (1/99.0)*val
	return avg

def grad_costfn(x,y,theta,m=99):
	mtx = np.zeros(3)
	for i in range(3):
		mtx[i] = (-1/m)*sum([((1.0 - h_thetai(y[k][0]*x[k,:],theta))*y[k][0]*x[k][i]) for k in range(m)])

	#grad0 = 1.0/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i]) for i in range(m)])
	

	return mtx	

def f(x1,theta):
	return (-(theta[0]+theta[1]*x1)/theta[2])

def construct_Hessian(x,y,theta,m=99):
	hess = np.zeros((3,3))
	for i in range(3):
		for j in range(3):
			hess[i][j] = (1/m)*sum([((1.0 - h_thetai(y[k][0]*x[k,:],theta))*x[k][i]*x[k][j]*h_thetai(y[k][0]*x[k,:],theta)) for k in range(m)])
	print hess
	return hess

def newtons_method(x,y,ep,max_iter):
	converged = False
	num_iter = 0

	#initialize theta 
	theta = np.zeros(3)

	J = costfn(x,y,theta)
	print ('Initial error is =',J)
	# start loop

	while not converged:
		#for i in range(3):
		#	sderi = 0
		#	dderi = 0
		#	for k in range(99):
		#		sderi += (1.0 - h_thetai(y[k][0]*x[k,:],theta))*y[k][0]*x[k][i]
		#		dderi += (1.0 - h_thetai(y[k][0]*x[k,:],theta))*h_thetai(y[k][0]*x[k,:],theta)*((x[k][i])**2)
		#	update = sderi/dderi
		#	theta[i] = theta[i] + update
		update = inv(construct_Hessian(x,y,theta)).dot(grad_costfn(x,y,theta))
		theta -= update
		Jnew = costfn(x,y,theta)
		if (abs(Jnew - J)<ep):
			converged = True
		num_iter += 1
		J = Jnew
		print ('Iteration number ',num_iter)

		if num_iter == max_iter:
			print 'Max Iterations Reached!'
			converged = True
	print ('Final error is =',J)
	return theta


if __name__ == '__main__':
	#Load the data
	x_train = np.loadtxt('logistic_x.txt')
	y_train = np.loadtxt('logistic_y.txt')
	y_train = y_train.reshape((99,1))
	#print y_train
	# Add intercept term
	icept = np.ones((99,1),dtype = 'float64')
	x_train = np.append(icept, x_train,axis=1)
	#print x_train.shape
	#print x_train[0][0]

	#plotting the data

	lab0 = np.argmax(y_train)
	x_trainn = x_train[0:lab0,1:3]
	x_trainp = x_train[lab0:,1:3]

	plt.plot(x_trainn[:,0],x_trainn[:,1],'rx')
	plt.plot(x_trainp[:,0],x_trainp[:,1],'go')
	plt.xlabel('x1')
	plt.ylabel('x2')
	#plt.show()

	max_iter = 1000
	ep = 0.000000000001

	theta = newtons_method(x_train,y_train,ep,max_iter)
	print theta.shape
	#print theta

	w = 0

	for k in range(99):
		if (((theta[0]+theta[1]*x_train[k][1]+theta[2]*x_train[k][2])*y_train[k][0])<0):
			w+=1
	w = w/99

	print'Training error is =',100*w

	print 'Done!'

	x1 = np.arange(0.0,10.0,0.1)
	plt.plot(x1,f(x1,theta))
	plt.show()




