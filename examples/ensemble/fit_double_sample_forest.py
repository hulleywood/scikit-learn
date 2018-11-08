"""
===================================================================
Fitting double sample forests on known data
===================================================================

TODO

"""
print(__doc__)

import numpy as np
from sklearn.ensemble import DoubleSampleForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

num_features = 6
num_observations = 5000
X = np.random.uniform(0, 1, size=(num_observations, num_features))
y = []
w = []

def func_x(x):
    return (1 + 1.0 / (1.0 + np.exp(-20.0 * (x - 1.0/3.0))))

for i in range(len(X)):
    wi = i % 2
    xi = X[i][wi]
    yi = func_x(xi)
    y.append(yi)
    w.append(wi)

y = np.array(y)
w = np.array(w)
tau = np.multiply(func_x(X[:,0]),func_x(X[:,1]))

X_train, X_test, y_train, y_test, w_train, w_test, tau_train, tau_test = train_test_split(
    X, y, w, tau, test_size=0.5, random_state=4)

regr = DoubleSampleForest(n_estimators=100, max_depth=30, random_state=2)
regr.fit(X_train, y_train, w_train)

y_pred = regr.predict_outcomes(X_test)
tau_pred = np.multiply(y_pred[:, 0], y_pred[:,1])
score = mean_squared_error(tau_test, tau_pred)

print(score)