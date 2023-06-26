#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing A Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[ : , 1:-1].values #stores the independant values of dataset into x
y = dataset.iloc[ : , -1].values #stores the dependant values into y

#Training the Linear Regression Model on the whole Dataset 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

#Training The polynomial Regression Model on whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 4) #Creates matrix of new features with their degrees
x_poly = poly_reg.fit_transform(x)
regressor_2 = LinearRegression()
regressor_2.fit(x_poly,y)


#Visualising the Linear Regression Results
plt.scatter(x,y, color = 'red')
plt.plot(x, regressor.predict(x), color ='blue')
plt.title('Truth Or Bluff(Linear regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising The polynomial regression model
plt.scatter(x,y, color = 'red')
plt.plot(x, regressor_2.predict(x_poly), color ='blue')
plt.title('Truth Or Bluff(Polynomial regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear regression
print(regressor.predict([[6.5]]))

#Predicting a new result with polynomial regression

print(regressor_2.predict(poly_reg.fit_transform([[6.5]])))

#Visualising The Polynomial Regression Results(for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y, color = 'red')
plt.plot(x_grid,regressor_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth Or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()