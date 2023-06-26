#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing A Dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[ : , :-1].values #stores the independant values of dataset into x
y = dataset.iloc[ : , -1].values #stores the dependant values into y 

#Taking care of Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer.fit(x[ : , 1:3])
x[ : , 1:3] = imputer.transform(x[: , 1:3])

#Encoding Categorial data
#ENCODING THE INDEPENDANT VARIABLE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder #convert our categories into numbers 

ct = ColumnTransformer(transformers=[('encoder' , OneHotEncoder(), [0])] , remainder='passthrough')
x = np.array(ct.fit_transform(x))

#ENCODING THE DEPENDANT VARIABLE
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)



#Splitting the dataset into Training Set and Test Set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x,y,test_size= 0.2 , random_state= 1)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[ : , 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:,3:] = sc.transform(x_test[:,3:])


print(x_train)
print(x_test)


