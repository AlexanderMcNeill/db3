import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Generate an un-balanced 2D dataset
np.random.seed(0)
X = np.vstack([np.random.normal(0, 1, (950, 2)),
               np.random.normal(-1.8, 0.8, (50, 2))])
y = np.hstack([np.zeros(950), np.ones(50)])

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='none',
            cmap=plt.cm.Accent);
plt.show()

X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5)
print X1.shape
print X2.shape

y2_pred = LogisticRegression().fit(X1, y1).predict(X2)
y1_pred = LogisticRegression().fit(X2, y2).predict(X1)
