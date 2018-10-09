import numpy as np
import pandas as pd
import sys
import pylab
import matplotlib.pyplot as plt
import math


def h_thetai(th,x,i):
	xint = np.append([1],x[i,0]).reshape((2,1))
	#print xint 
	val = np.inner(np.transpose(xint),np.transpose(th))[0,0]
	return val

def compute_cost_function(m,th,x,y):
	sm = 0
	#ctr = 0
	for i in range(m):
		sm = sm + (h_thetai(th,x,i) - y[i,0])**2
	#print 0.5*sm
	return 0.5*sm


def gradient_descent(alpha, x, y, ep, max_iter):
	converged = False
	num_iter = 0
	# Number of training samples
	m = x.shape[0]
	# Intialize theta vector to vector of zeroes
	th = np.zeros((2,1), dtype = np.float128)

	#Error , J(theta)
	J = compute_cost_function(m,th,x,y)
	print ('Initial Error is =',J)
	while not converged:
		#Compute iterations for each of the parameters in tandem
		grad0 = sum([h_thetai(th,x,i)-y[i,0] for i in range(m)])
		grad1 = sum([(h_thetai(th,x,i)-y[i,0])*x[i,0] for i in range(m)])

		temp0 = th[0,0] - alpha*grad0
		temp1  = th[1,0] - alpha*grad1

		if (abs(temp0-th[0,0]) < ep and abs(temp1-th[1,0]) < ep):
			converged = True

		th[0,0] = temp0
		th[1,0] = temp1

		J = compute_cost_function(m,th,x,y)
		#print J
		num_iter += 1
		print ("Iteration number",num_iter)

		if num_iter == max_iter:
			print('Max iterations reached')
			converged = True

	return th 



if __name__ == '__main__':

	df = pd.read_csv('./train.csv', names = ['x','y'], skiprows = 1)
	#print df
	#plt.plot(df['x'],df['y'],'ro')
	x = df['x']
	y = df['y']
	x = np.asarray(x,dtype = np.float128)
	x = x.reshape((700,1))
	y = np.asarray(y,dtype = np.float128)
	y = y.reshape((700,1))

	plt.plot(x,y,'ro')
	plt.ylabel('Output')
	plt.xlabel('Input')
	plt.savefig('Input')


	ep =  0.00001
	alpha = 0.00000001
	max_iter = 1000
	print ('Start  learning')
	th  = gradient_descent(alpha,x,y,ep,max_iter)
	print ('theta 0 is = ',th[0,0])
	print ('theta 1 is = ', th[1,0])

	y_predict = np.zeros((700,1))

	for i in  range(x.shape[0]):
		y_predict[i,0] = th[0,0] +  th[1,0]*x[i,0]

	pylab.plot(x,y,'o')
	pylab.plot(x,y_predict,'k-')
	pylab.show()
	print ('Done!')

