"""
===================================================================
Plotting double sample forests on simulated data
===================================================================

This example plots the mean squared error of a DoubleSampleForest trained
and evaluated on simulated data. By using the first two features to create the
treatment effect, we're able to visualize the ability of the model to learn which
features contribute to the treatment effect by varying the total number of
features in the dataset. This example also serves to demonstrate that model
performance improves by increasing the number of observations and number of trees.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import DoubleSampleForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def generate_data(n_features, n_samples):
    X = np.random.uniform(0, 1, size=(n_samples, n_features))
    w = np.random.binomial(1, 0.5, size=n_samples)

    def func_x(x):
        return (1 + 1.0 / (1.0 + np.exp(-20.0 * (x - 1.0 / 3.0))))

    tau = func_x(X[:, 0]) * func_x(X[:, 1])
    y = ((w - 0.5) * tau) + np.random.randn(n_samples)

    return X, y, w, tau

def train_and_score(n_features, n_samples, n_trees):
    X, y, w, tau = generate_data(n_features, n_samples)
    X_train, X_test, y_train, y_test, w_train, w_test, tau_train, tau_test = train_test_split(
        X, y, w, tau, test_size=0.5, random_state=4)

    regr = DoubleSampleForest(n_estimators=n_trees, random_state=2, n_jobs=-1)
    regr.fit(X_train, y_train, w_train)
    tau_pred = regr.predict_effect(X_test)

    return mean_squared_error(tau_pred, tau_test)

n_features = [2, 5, 10, 20]
n_samples = [1000, 25000, 50000, 75000, 100000]
n_trees = [20, 50, 100]
subplot_num = 0

for t in n_trees:
    subplot_num += 1
    plt.subplot(1, len(n_trees), subplot_num)
    plt.title('n_trees = {}'.format(t))
    plt.ylim(0, 1)
    if subplot_num == 1:
        plt.ylabel('mse')
        plt.xlabel('n_samples')

    for f in n_features:
        mse = []
        for s in n_samples:
            mse.append(train_and_score(f, s, t))
            print('Finished with: {}, {}, {}'.format(t, f, s))

        plt.plot(n_samples, mse, label=f)


plt.legend(loc='upper left', title='n_features')
plt.tight_layout()
plt.show()