# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 13:06:42 2015

@author: mcnear1
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

"""Exercise 1 - K Means"""

print("========================Exercise 1 - K Means========================")

# Creating artificial data
X, y = make_blobs(n_samples=300, centers=3, random_state=0, cluster_std=0.60)

# Displaying what the artificial data looks like
plt.title("Artificial data")
plt.scatter(X[:, 0], X[:, 1], s=50);
plt.show()

# Creating k means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=8)
# Creating k means that fits the data and getting the labels for each point
y_hat = kmeans.fit(X).labels_

# Displaying the clustering that the kmeans has defined
plt.title("Groupings created by k-Means")
plt.scatter(X[:, 0], X[:, 1], c=y_hat)
plt.show()

print("\n========================Exercise 2 - K-Nearest neighbors========================\n")
# Reading in the data from the file
vs = np.genfromtxt("video_store_2.csv", delimiter=",", names=True, dtype=(int, "|S1", float, int, int, float, "|S10", "|S3"))

# Getting the data I will be using
vs_records = vs[['Gender','Income','Age','Rentals','Avg_Per_Visit','Genre']]
vs_names = vs_records.dtype.names
vs_dict = [dict(zip(vs_names, record)) for record in vs_records]
vs_vec = DictVectorizer()
X = vs_vec.fit_transform(vs_dict).toarray()
y = vs['Incidentals']

# Defining where the data is split for the training set and test set
tpercent = 0.8
tsize = tpercent * len(X)

# Splitting the data into training and testing sets
X_train = X[:tsize,:]
X_test = X[tsize:,:]

y_train = y[:tsize]
y_test = y[tsize:]

# Normalising all the data
min_max_scaler = preprocessing.MinMaxScaler()
X_train_norm = min_max_scaler.fit_transform(X_train)
X_test_norm = min_max_scaler.fit_transform(X_test)

# Creating the n K-Nearest neightbor model
knn = KNeighborsClassifier(n_neighbors=5)

# Fitting the model to the video store training data
knn.fit(X_train_norm, y_train)

print("Knn model created")

print("\n========================Exercise 3 - Error analysis: Precision and recall========================\n")

# Predicting the results with the K-Nearest neighbor model
y_predict = knn.predict(X_test_norm)

# Displaying report on how well it has performed
print("Classification report for predictions on video store data set using K-Means")
print metrics.classification_report(y_test, y_predict)

print("\n========================Exercise 4 - Support vector machines========================\n")

# Creating a support vector machine model
clf = SVC(kernel='rbf')
# Fitting the model to the video store training data
clf.fit(X_train_norm, y_train)
# Predicting the results of the test data using the model
y_predict = clf.predict(X_test_norm)

# Displaying a report on how well the support vector machines model worked
print("Classification report for predictions on video store data set using Vector machines")
print metrics.classification_report(y_test, y_predict)

print("\n========================Exercise 5 - Logistic regression========================\n")
# Creating a logistic regression model
logreg = LogisticRegression()
# Fitting the model to my data
logreg.fit(X_train_norm, y_train)
# Predicting the results of all of the test data
y_predict = logreg.predict(X_test_norm)

# Displaying a report on how well the logistic regression model performed
print("Classification report for predictions on video store data set using logistic regession")
print metrics.classification_report(y_test, y_predict)

print("\n========================Exercise 6 - Principal Component Analysis========================\n")
# Todo: Add principle component analysis

print("========================Exercise 7 - Artificial neural networks========================\n"
+ "\nFor the network described in ann.png id the input vector was (1,0) "
+ "\nthe result would be 10 what would map to be very close to one."
+ "\n1 * -10 + 1 * 20 + 0 * 20 = 10 ≈ 1")

print("\n========================Exercise 8 - Validation========================\n")

"""
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

// Loading the data
digits = load_digits()
X = digits.data
y = digits.target

// Generating cross value score
cv = cross_val_score(KNeighborsClassifier(1), X, y, cv=10)
cv.mean()
"""

print("This code is using cross Validation to ensure that the model isn't going to over fit the data."
+"\nIn this example they are they are using the KNN algorithm and cross validating 10 times."
+"\nCross validating splits the data into test and train groups and gets the average accuracy"
+"\nof all of the splits a.k.a folding the data."
+"\nThis average is what is being output on the final line of the code.")

print("\n========================Exercise 9 - Regularization========================\n")

"""
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn import linear_model

// Loading the data
boston = load_boston()

// Splitting the data into training and test sets
// Half the data for testing and half for training
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.5, random_state=0)

// Creating a linear regression model and printing the coefficients
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train);
print regr.coef_

print "-------------------"

// Creating another linear regression model using regression and printing the coefficients
ridgeRegr = linear_model.SGDRegressor(loss='squared_loss', penalty='l2') #Hint: The answer to the 1st question lies here
ridgeRegr = linear_model.Ridge(alpha = 10000)
ridgeRegr.fit(X_train, y_train)
print ridgeRegr.coef_
"""

print("This code is showing the difference between using regularization and not using regularization."
+"\nThe second model imposes a penalty upon each of the coefficients bringing them closer to zero,
+"\nthis is why the values are different."
+"\nSquared weights penalizes large values more this is why the values in the second model are much smaller."
)
