# -*- coding: utf-8 -*-
"""
Created on Tue Sep 08 16:28:49 2015

@author: mcnear1
"""

from IPython.core.display import Image, display
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

display(Image(filename='images/iris_setosa.jpg'))
print "Iris Setosa\n"

display(Image(filename='images/iris_versicolor.jpg'))
print "Iris Versicolor\n"

display(Image(filename='images/iris_virginica.jpg'))
print "Iris Virginica"

'''
If we want to design an algorithm to recognize iris species, what might the features be? What might the labels be?
Remember: we need a 2D data array of size [n_samples x n_features], and a 1D label array of size n_samples.
What would the n_samples refer to? Each of the flowers we are analysing
What might the n_features refer to? The features of each flower
Remember that there must be a fixed number of features for each sample, and feature number i must be a similar kind of quantity for each sample.
'''

#Loads data with 150 samples, each having 4 features (sepal length in cm, sepal width in cm, petal length in cm, and petal width in cm)
iris = load_iris()

print iris.keys()

# Getting how many samples and features there are
n_samples, n_features = iris.data.shape
print (n_samples, n_features)

# Printing the features of the first sample
print iris.data[0]

'''
How would you print the 2nd instance? iris.data[1]
How would you print the last instance? iris.data[-1]
How would you print the last instance if you don't know the length of the data set? iris.data[-1]
'''

# The shape of the data and its target type
print iris.data.shape
print iris.target.shape

#The first samples data and its taget type
print iris.data[0]
print iris.target[0]

#printing all the targets for the data
print iris.target

# Printing the names of the targets
print iris.target_names


def plot_iris_projection(x_index, y_index):
    # this formatter will label the colorbar with the correct target names
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

    plt.scatter(iris.data[:, x_index], iris.data[:, y_index],
                c=iris.target)
                
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(iris.feature_names[x_index])
    plt.ylabel(iris.feature_names[y_index])
    plt.show()
    
#Calling the method with the index of the 2 features we want to compare
plot_iris_projection(2, 3)

'''
Change x_index and y_indexin the above script and try to estimate different 
combination of just two features which better separate the three classes or irises.
'''
print iris.feature_names
plot_iris_projection(2, 0)

from sklearn.datasets import load_digits
digits = load_digits()

print digits.keys()

n_samples, n_features = digits.data.shape
print (n_samples, n_features)

# Printing the features of the first sample
print digits.data[0]
#Printing all of the targets
print digits.target

