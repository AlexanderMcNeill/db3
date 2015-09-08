# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:31:50 2015

@author: mcnear1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import linear_model

data = pd.read_csv('Advertising.csv', index_col=0)
feature_cols = ['TV', 'Radio', 'Newspaper']

# Question 1
plt.scatter(data['TV'], data['Sales'])
plt.xlabel('TV ads')
plt.ylabel('Sales');
plt.title('Question 1: Expenditure on TV ads against Sales');
plt.show()

# Question 2
plt.scatter(data['Radio'], data['Sales'])
plt.xlabel('Radio')
plt.ylabel('Sales');
plt.title('Question 2: Expenditure on Radio against Sales');
plt.show()

# Question 3
plt.scatter(data['Newspaper'], data['Sales'])
plt.xlabel('Newspaper')
plt.ylabel('Sales');
plt.title('Question 3: Expenditure on Newspaper against Sales');
plt.show()

# Question 4: Are the TV and radio features positively correlated to sales?
print("Question 4: Both the TV and Radio features have a moderate positive corrolation to sales")

# Question 5: Is the Newspaper features positively or negatively correlated with sales
print("\nQuestion 5: The newspaper feature has a no correlation or very weak positive correlation to sales")

# Question 6: Split the data in two halfs: training set and test set
x = np.array(data[feature_cols])
y = np.array(data['Sales'])

x, y = shuffle(x, y, random_state=1)

train_set_size = len(x) / 2

x_train = x[:train_set_size]
x_test = x[train_set_size:]

y_train = y[:train_set_size]
y_test = y[train_set_size:]

# Question 7: Fit a multivariate linear regression model on the training data using all the features available
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train);

# Question 8: What are the intercepts and coefficients of the model?
print("\nQuestion 8: coeffefficients and intercept")
print("The coeffefficients for the regression model: " + str(regr.coef_))
print("The intercept for the regression model: " + str(regr.intercept_))

# Question 9: What is the R2 score?
print("\nQuestion 9: R2 Scores")
print("Training data R2 score: " + str(regr.score(x_train, y_train))) # R2 score for the training data
print("Testing data R2 score: " + str(regr.score(x_test, y_test))) # R2 score for the test data

# Question 10: Predict Results
print("\nQuestion 10: Predict Results")
option_one = [200,30,70]
option_two = [200,90,10]
option_three = [100,100,100]

option_one.append(regr.predict(option_one))
option_two.append(regr.predict(option_two))
option_three.append(regr.predict(option_three))

output_string = "For TV: ${}, Radio: ${} and Newspaper: ${} the estimated Sales are: {}"
print(output_string.format(option_one[0], option_one[1], option_one[2], option_one[3]))
print(output_string.format(option_two[0], option_two[1], option_two[2], option_two[3]))
print(output_string.format(option_three[0], option_three[1], option_three[2], option_three[3]))