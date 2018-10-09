import numpy as np



a = np.ones((2,2))
print a
print a[0,:].shape

print np.inner(np.transpose(a[0,:]),np.transpose(a[1,:]))

print 3*a[0,:]
print np.zeros(3)
x = np.empty([2,2])
#print x[:,0]
print x 

#x[i][j] = i+j for i,j in range(2)
#print x
x = np.ones((2,1))
print a.dot(x)