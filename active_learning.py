from scipy.stats import entropy
import numpy as np


class ActiveLearner():
    uncertainty_sampling_methods = [
        'least_confident',
        'max_margin',
    ]

    query_by_committee_methods = [
        'vote_entropy',
        'kl_divergence',
    ]

    def __init__(self, query_type='least_confident'):
        self.query_type = query_type

    def rank(self, clf, X_unlabeled, num_queries=None):
        if num_queries is None:
            num_queries = X_unlabeled.shape[0]

        elif type(num_queries) == float:
            num_queries = map(int, num_queries * X_unlabeled.shape[0])

        if self.query_type in self.uncertainty_sampling_methods:
            scores = self.do_uncertainty_sampling(clf, X_unlabeled)

        elif self.query_type in self.query_by_committee_methods:
            scores = self.do_query_by_committee(clf, X_unlabeled)

        else:
            # if self.query_type == 'diversity_sampling':
            #     pass
            raise NotImplementedError

        rankings = np.argsort(-scores)[:num_queries]
        return rankings

    def do_uncertainty_sampling(self, clf, X_unlabeled):
        probs = clf.predict_proba(X_unlabeled)

        if self.query_type == 'least_confident':
            return 1 - np.amax(probs, axis=1)

        elif self.query_type == 'max_margin':
            margin = np.partition(-probs, 1, axis=1)
            return -np.abs(margin[:, 0] - margin[:, 1])

    def do_query_by_committee(self, clf, X_unlabeled):
        num_classes = len(clf[0].classes_)
        num_members = len(clf)
        preds = []

        if self.query_type == 'vote_entropy':
            for model in clf:
                y_out = model.predict(X_unlabeled)
                preds.append(np.eye(num_classes)[y_out])

            votes = np.apply_along_axis(np.sum, 0, np.stack(preds))/num_members
            return np.apply_along_axis(entropy, 1, votes)

        elif self.query_type == 'kl_divergence':
            for model in clf:
                preds.append(model.predict_proba(X_unlabeled))

            decision = np.mean(np.stack(preds), axis=0)
            divergence = []
            for y_out in preds:
                divergence.append(entropy(decision.T, y_out.T))

            return np.apply_along_axis(np.mean, 0, np.stack(divergence))
