# -*- coding: utf-8 -*-
"""
Created on Tue Sep 08 16:14:31 2015

@author: mcnear1

note: Alot of this code was copied due to it being a intro to the libraries
"""

import numpy as np

# Generating a random Matrix of dimensions 3×5
X = np.random.random((3,5))
print X


# Accessing elements

# get a single element
print X[0, 0]

# get a row
print X[1]

# get a column
print X[:, 1]

# Transposing an array
print X.T

# Turning a row vector into a column vector
y = np.linspace(0, 12, 5) # Making a row vector that has 5 slots
print y

# make into a column vector
print y[:, np.newaxis]


#Matplotlib

import matplotlib.pyplot as plt

# plotting a line
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x));
plt.show();

# scatter-plot points
x = np.random.normal(size=500)
y = np.random.normal(size=500)
plt.scatter(x, y);
plt.show();

# showing images
x = np.linspace(1, 12, 100)
y = x[:, np.newaxis]

im = y * np.sin(x) * np.cos(y)
print(im.shape)

# imshow - note that origin is at the top-left!
plt.imshow(im);
plt.show()

# Contour plot - note that origin here is at the bottom-left!
plt.contour(im);
plt.show()

# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
mu, sigma = 2, 0.5
v = np.random.normal(mu,sigma,10000)


import pylab

# Plot a normalized histogram with 50 bins
pylab.hist(v, bins=50, normed=1)       # matplotlib version (plot)
pylab.show()

# Compute the histogram with numpy and then plot it
(n, bins) = np.histogram(v, bins=50, normed=True)  # NumPy version (no plot)
pylab.plot(.5*(bins[1:]+bins[:-1]), n)
pylab.show()