import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from active_learning.active_learning import ActiveLearner
from collections import Counter

X, y = load_iris(return_X_y=True)
n_queries = 100

X_labeled, y_labeled = X[[0, 50, 100]], y[[0, 50, 100]]
estimators = [LogisticRegression(solver='lbfgs', multi_class='auto'), LogisticRegression(solver='lbfgs', multi_class='auto')]

for estimator in estimators:
    estimator.fit(X_labeled, y_labeled)

learner = ActiveLearner(strategy='vote_entropy')

preds = []
results = []
correct = 0

for _ in range(n_queries):
    query_idx = learner.rank(estimators, X, num_queries=1)
    X_labeled = np.concatenate((X_labeled, X[query_idx]), axis=0)
    y_labeled = np.concatenate((y_labeled, y[query_idx]), axis=0)
    for estimator in estimators:
        estimator.fit(X_labeled, y_labeled)
        preds.append(estimator.predict(X))
    majority_votes = np.apply_along_axis(lambda x: Counter(x).most_common()[0][0], 0, np.stack(preds))
    accuracy = np.sum(majority_votes == y) / np.shape(X)[0]
    correct += np.sum(majority_votes == y)
    results.append(accuracy)
    print('Accuracy:', accuracy)
print(results)
accuracy = np.sum(majority_votes == y) / np.shape(X)[0]
print('Accuracy:', accuracy)
