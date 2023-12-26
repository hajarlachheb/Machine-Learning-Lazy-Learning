import numpy as np


def euclidean(point, data):
    """Euclidean distance between a point  & data"""
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


def most_common(lst):
    """Returns the most common element in a list"""
    return max(set(lst), key=lst.count)


class KNeighborsClassifier:
    def __init__(self, k, dist_metric, voting_metric, weights):
        self.y_train = None
        self.X_train = None
        self.k = k
        self.dist_metric = dist_metric
        self.voting_metric = voting_metric
        self.weights = weights

    def fit(self, X_train, y_train):
        """
        Fit training data
        :param X_train: training features
        :param y_train: training class labels
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predict labels for a given test data
        :param X_test: test data features
        :return: predicted labels for each instance
        """
        pred_labels = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train, self.weights)  # Compute distances
            k_labels, k_distances = self.get_neighbors(distances)  # Get nearest neighbors
            y_pred = self.voting_metric(k_labels, k_distances)  # Vote for labels and get prediction
            pred_labels.append(int(y_pred))
        return pred_labels

    def evaluate(self, X_test, y_test):
        """
        Predict and evaluate results from classification
        :param X_test: test data features
        :param y_test: test data class labels
        :return: accuracy, number of correct predictions and number of incorrect predictions
        """
        y_pred = self.predict(X_test)
        correct_pred = sum(y_pred == y_test)
        incorrect_pred = sum(y_pred != y_test)
        accuracy = correct_pred / len(y_test)
        return accuracy, correct_pred, incorrect_pred

    def get_neighbors(self, distances):
        """
        Get the nearest neighbors (the closest points) from given distances
        :param distances: distances from instance to test data points
        :return: labels and distances to the k-nearest neighbors
        """
        sorted_indices = np.argsort(distances)
        k_y = self.y_train[sorted_indices[:self.k]]
        k_dist = np.array(distances)[sorted_indices[:self.k]]
        return k_y, k_dist

    def find_kneighbors(self, X):
        """
        Find distances to neighbors from a given set of points.
        :param X: (array-like) instances (features values) for which to find neighbors
        :return: sorted distances to the k-nearest neighbors
        """
        for instance in X:
            distances = self.dist_metric(instance, self.X_train, self.weights)  # Compute distances
            labels, dist = self.get_neighbors(distances)
            return dist

