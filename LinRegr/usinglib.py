from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./train.csv', names=['x','y'], skiprows = 1)
x = pd.DataFrame(df, columns=['x'])
y = pd.DataFrame(df, columns=['y'])
x = np.asarray(x,dtype = np.float128)
x = x.reshape((700,1))
y = np.asarray(y,dtype = np.float128)
y = y.reshape((700,1))
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(x, y)
# Make predictions using the testing set
y_pred = regr.predict(x)
#print('y pred', y_pred);
plt.scatter(x, y,  color='black')
plt.plot(x, y_pred, color='blue', linewidth=1)
plt.show()