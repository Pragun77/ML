#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing A Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[ : , 1:-1].values #stores the independant values of dataset into x
y = dataset.iloc[ : , -1].values #stores the dependant values into y

y  = y.reshape(len(y), 1) #converting into 2-d array

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x) 
y = sc_y.fit_transform(y) #converts the values in a range of -3 to +3


#training the svr model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel= 'rbf')
regressor.fit(x,y)

#Predicting the new result
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1)))


#Visualising The SVR Results
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)), color ='blue')
plt.title('Truth Or Bluff(SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising The SVR Results(for higher resolution and smoother curve)
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid,sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth Or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()