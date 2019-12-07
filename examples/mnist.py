from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from copy import deepcopy
from scipy.stats import entropy
from scipy import spatial
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


def dist(xy1, xy2):
    P = np.add.outer(np.sum(xy1**2, axis=1), np.sum(xy2**2, axis=1))
    N = np.dot(xy1, xy2.T)
    dists = np.sqrt(P - 2*N)
    return dists

def find_min_dist(xy1, xy2):
    tree = spatial.cKDTree(xy2)
    mindist, minid = tree.query(xy1)
    return mindist, minid

def budget_labeling(X_candidates, X_unlabeled):
    # Choose the node minimizes the overall
    # distance between labeled and unlabeled nodes
    # Keep doing this until convergence / no change
    min_dist = 1000000000
    argmin = None
    for candidate in X_candidates:
        distance, _ = find_min_dist(np.array([candidate]), X_unlabeled).item()
        if min_dist > distance:
            min_dist = distance
            argmin = candidate
    if argmin is None:
        pass
    else:
        return argmin


def plot_graph(num_samples, random_sampling_list, all_query_list, label_list, title_name, fig_name):
    sns.set_style("darkgrid")
    plt.plot(
        num_samples, np.mean(random_sampling_list, axis=0), 'red',
        num_samples, np.mean(all_query_list[label_list[1]], axis=0), 'blue',
        num_samples, np.mean(all_query_list[label_list[2]], axis=0), 'green',
    )
    plt.legend([i.replace('_', ' ').capitalize() for i in label_list], loc=3)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Queries')
    plt.title(title_name)
    plt.ylim([0,1])
    plt.savefig(fig_name)


def uncertainty_sampling(type_of_learning='pool', limited_budget=False):
    results_holder = {
        'least_confident': [],
        'max_margin': [],
    }
    all_uncertainty_sampling_results = deepcopy(results_holder)
    all_random_sampling_results = []
    # Get the MNIST Data
    mnist = fetch_mldata('MNIST original')
    clf = [LogisticRegression(solver='lbfgs', multi_class='auto') for _ in range(3)]
    # from sklearn.linear_model import LinearRegression
    # clf = [LinearRegression() for _ in range(3)]

    for iteration in range(10):
        print ('Iteration #', iteration)
        uncertainty_sampling_results = deepcopy(results_holder)
        num_samples, random_sampling_results = [], []

        X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.5)

        X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(X_train, y_train, test_size=0.999)

        query_list = [10*i for i in range(11)]
        if type_of_learning == 'stream':
            query_list = [1 for i in range(100)]
        for query_number, num_queries in enumerate(query_list):
            print ('Query #', query_number, 'of', len(query_list))
            num_samples.append(num_queries)

            # For random sampling, select random queries
            random_queries = np.random.choice(X_unlabeled.shape[0], num_queries, replace=False)

            # Let oracle label randomly chosen points (very naive strategy, but works
            # surprisingly well!!)
            if limited_budget == True:
                random_queries = budget_labeling(random_queries, X_unlabeled)
            X_post_query = np.concatenate((X_labeled, X_unlabeled[random_queries]))
            y_post_query = np.concatenate((y_labeled, y_oracle[random_queries]))

            # Train post labeling
            clf[0].fit(X_post_query, y_post_query)

            # Measure accuracy
            accuracy = np.sum(clf[0].predict(X_test) == y_test) / np.shape(X_test)[0]
            random_sampling_results.append(accuracy)

            print('Random Sampling : Accuracy', accuracy)

            j = 1
            for query_type in uncertainty_sampling_results:
                # First fit the existing labeled data
                clf[j].fit(X_labeled, y_labeled)

                # Create the Active Learner
                al = ActiveLearner(query_type=query_type)

                # Rank according to query_type for the oracle to label
                idx = al.rank(clf[j], X_unlabeled, num_queries)

                # Oracle labels the relevant indices
                if limited_budget == True:
                    idx = budget_labeling(idx, X_unlabeled)
                X_post_query = np.concatenate((X_labeled, X_unlabeled[idx]))
                y_post_query = np.concatenate((y_labeled, y_oracle[idx]))

                # Train the model post labeling
                clf[j].fit(X_post_query, y_post_query)

                # Measure accuracy
                accuracy = np.sum(clf[j].predict(X_test) == y_test) / np.shape(X_test)[0]
                uncertainty_sampling_results[query_type].append(accuracy)
                print(query_type, ': Accuracy', accuracy)
                j+=1

        all_random_sampling_results.append(random_sampling_results)
        for query_type in uncertainty_sampling_results:
            all_uncertainty_sampling_results[query_type].append(
                uncertainty_sampling_results[query_type])

    plot_graph(num_samples, all_random_sampling_results, all_uncertainty_sampling_results,
               ['random_sampling', 'least_confident', 'max_margin'], 'MNIST - Uncertainty Sampling', 'mnist_uncertainty_sampling.jpg')

def qbc(type_of_learning='pool', limited_budget=False):
    results_holder = {
        'kl_divergence': [],
        'vote_entropy': [],
    }
    all_query_by_commitee_results = deepcopy(results_holder)
    all_random_sampling_results = []
    # Get the MNIST Data
    mnist = fetch_mldata('MNIST original')
    clf=[LogisticRegression(solver='lbfgs', multi_class='auto') for _ in range(10)]
    # from sklearn.linear_model import LinearRegression
    # clf = [LinearRegression() for _ in range(10)]

    for iteration in range(2):
        print('Iteration #', iteration)
        query_by_commitee_results = deepcopy(results_holder)
        num_samples, random_sampling_results = [], []

        X_train, X_test, y_train, y_test = train_test_split(
            mnist.data, mnist.target, test_size=0.5)

        X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(X_train, y_train, test_size=0.999)

        query_list = [10*i for i in range(11)]
        if type_of_learning == 'stream':
            query_list = [1 for i in range(100)]
        for query_number, num_queries in enumerate(query_list):
            print ('Query #', query_number, 'of', len(query_list))
            num_samples.append(num_queries)
            # For random / diversity sampling
            random_queries = np.random.choice(X_unlabeled.shape[0], num_queries, replace=False)
            if limited_budget == True:
                random_queries = budget_labeling(random_queries, X_unlabeled)
            X_post_query = np.concatenate((X_labeled, X_unlabeled[random_queries]))
            y_post_query = np.concatenate((y_labeled, y_oracle[random_queries]))
            preds = []
            for model in clf:
                # Train all the committee members post labelling
                model.fit(X_post_query, y_post_query)
                preds.append(model.predict(X_test))

            # Give result based on majority vote
            majority_votes = np.apply_along_axis(lambda x: Counter(x).most_common()[0][0], 0, np.stack(preds))
            accuracy = np.sum(majority_votes == y_test) / np.shape(X_test)[0]
            random_sampling_results.append(accuracy)
            print('Diversity Sampling - Accuracy :', accuracy)

            # For other query strategies
            for query_type in query_by_commitee_results:
                for model in clf:
                    model.fit(X_labeled, y_labeled)

                al = ActiveLearner(query_type=query_type)
                for model in clf:
                    model.classes_ = np.arange(10)
                idx = al.rank(clf, X_unlabeled, num_queries)
                if limited_budget == True:
                    idx = budget_labeling(idx, X_unlabeled)
                X_post_query = np.concatenate((X_labeled, X_unlabeled[idx]))
                y_post_query = np.concatenate((y_labeled, y_oracle[idx]))

                preds = []
                for model in clf:
                    model.fit(X_post_query, y_post_query)
                    preds.append(model.predict(X_test))

                majority_votes = np.apply_along_axis(lambda x: Counter(x).most_common()[0][0], 0, np.stack(preds))
                accuracy = np.sum(majority_votes == y_test) / np.shape(X_test)[0]
                query_by_commitee_results[query_type].append(accuracy)
                print(query_type, '- Accuracy :', accuracy)

        all_random_sampling_results.append(random_sampling_results)
        for query_type in query_by_commitee_results:
            all_query_by_commitee_results[query_type].append(
                query_by_commitee_results[query_type])

    plot_graph(num_samples, all_random_sampling_results, all_query_by_commitee_results,
               ['random_sampling', 'kl_divergence', 'vote_entropy'], 'MNIST - Query By Commitee', 'mnist_query_by_commitee.jpg')

if __name__ == '__main__':
    uncertainty_sampling('pool', limited_budget=False)
    # uncertainty_sampling('stream')
    qbc('pool', limited_budget=False)
