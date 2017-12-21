import pickle
import numpy as np
import math

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        trees = []
        effects = []
        num_samples, num_features = np.shape(X)
        weights = np.ones(num_samples)/num_samples
        class_dist = np.zeros(num_samples)
        for i in range(self.n_weakers_limit):
            clf = self.weak_classifier
            best_tree = clf.fit(X, y, weights)
            y_pre = clf.predict(X)
            precision = np.mean((y == y_pre))
            error = 1-precision
            alpha = 0.5*math.log(1/(error+1e-8)-1)
            trees.append(best_tree)
            effects.append(alpha)
            exp_factor = -alpha * y * y_pre
            weights = weights * np.exp(exp_factor)
            weights = weights / weights.sum()
            class_dist += alpha * y_pre
            sum_precision = np.mean((y == np.sign(class_dist)))
            print(i+1, precision, sum_precision)
        self.save(trees, './trees')
        self.save(effects, './effects')
        pass

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        s, f = np.shape(X)
        scores = np.zeros(s)
        trees = self.load('./trees')
        effects = self.load('./effects')
        for i in range(self.n_weakers_limit):
            tree = trees[i]
            v = tree.predict(X)
            scores += v * effects[i]
        return scores
        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        scores = self.predict_scores(X)
        labels = np.sign(scores-threshold)
        return labels
        pass

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
