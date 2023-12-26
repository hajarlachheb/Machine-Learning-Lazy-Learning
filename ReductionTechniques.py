import KNN
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_X_y
from six.moves import xrange


class CNN:
    """
    Condensed Nearest Neighbors technique for instance reduction.
    A classifier (k-NN model) must be given in order to perform instance reduction.
    """

    def __init__(self, classifier):
        self.y_ = None
        self.X_ = None
        self.classifier = classifier
        self.cnn_ = set()

    def get_indices(self, X, y):
        """
        Get indices from the subset obtained after performed instance reduction with CNN.
        :param X: Features from training data (np.array-like of size nxm where n: num instances; m: num of features)
        :param y: Classes from training instances (np.array-like of size n where n: num instances)
        :return: indices from the reduced training data
        """
        self.X_, self.y_ = X, y
        n, d = X.shape
        additions = True
        self.cnn_ = {0}

        while additions:
            additions = False
            # Loop over all instances
            for i in xrange(1, n):
                # Store instance features and class
                xi, yi = X[i, :], y[i]
                # Create a list with the CNN indices
                cnn_arr = np.array(list(self.cnn_))
                # Select the reduced data
                cnnX = X[cnn_arr, :]
                cnnY = y[cnn_arr]
                # Fit the k-NN model with the reduced data
                self.classifier.fit(cnnX, cnnY)
                # Predict outputs
                yi_pred = self.classifier.predict(xi.reshape(1, -1))

                # Check if there is label in training data different than the obtained as prediction
                if yi != yi_pred:
                    if i not in self.cnn_:  # Check if index is already stored
                        self.cnn_.add(i)  # Add index into CNN list
                        additions = True  # Keep running the 'while' loop
        indices = np.array(list(self.cnn_))
        return indices


class RNN:
    """
    Reduced Nearest Neighbors technique for instance reduction.
    A classifier (k-NN model) must be given in order to perform instance reduction.
    """

    def __init__(self, classifier):
        self.reduction_ = None
        self.reduced_y_ = None
        self.reduced_X_ = None
        self.y_ = None
        self.X_ = None
        self.rnn_ = None
        self.cnn_ = set()
        self.classifier = classifier

    def reduce_data(self, X, y):
         """
        Reduce data with RNN technique
        :param X: Features from training data (np.array-like of size nxm where n: num instances; m: num of features)
        :param y: Classes from training instances (np.array-like of size n where n: num instances)
        :return: reduced training data (X and y)
        """
        self.X_, self.y_ = X, y

        cnn = CNN(self.classifier)
        self.rnn_ = set(cnn.get_indices(self.X_, self.y_))

        for i in self.rnn_:
            rnn_candidate = self.rnn_ - {i}
            rnn_arr = np.array(list(rnn_candidate))
            rnnX, rnnY = X[rnn_arr, :], y[rnn_arr]

            self.classifier.fit(rnnX, rnnY)
            ypred = self.classifier.predict(X)
            if sum(ypred == y) == len(y):
                self.rnn_ = rnn_candidate

        rnn_array = np.array(list(self.rnn_))
        self.reduced_X_ = self.X_[rnn_array, :]
        self.reduced_y_ = self.y_[rnn_array]
        self.reduction_ = (1.0 - float(len(self.reduced_y_)) / len(self.y_)) * 100

        return self.reduced_X_, self.reduced_y_, self.reduction_


class RENN:
    """
    Repeated Edited Nearest Neighbors technique for instance reduction.
    Run a 'num-iter' times the ENN reduction technique for instance reduction.
    A classifier (k-NN model) must be given in order to perform instance reduction.
    """

    def __init__(self, classifier, num_iter):
        self.y_ = None
        self.X_ = None
        self.reduced_X_ = None
        self.reduced_y_ = None
        self.reduction_ = None
        self.classifier = classifier
        self.num_iter = num_iter

    def reduce_data(self, X, y):
        """
        Reduce data with RENN technique
        :param X: Features from training data (np.array-like of size nxm where n: num instances; m: num of features)
        :param y: Classes from training instances (np.array-like of size n where n: num instances)
        :return: reduced training data (X and y)
        """
        self.X_ = X
        self.y_ = y

        # Create an ENN instance
        enn = ENN(self.classifier)

        # First assignment to match inside-loop
        red_X = self.X_
        red_y = self.y_

        # Loop for a given number of iterations
        for i in range(self.num_iter):
            red_X, red_y = enn.reduce_data(red_X, red_y)  # Run ENN algorithm

        # Store the final reduced data
        self.reduced_X_ = red_X
        self.reduced_y_ = red_y

        # Check reduction (in terms of storage)
        self.reduction_ = (1.0 - float(len(self.reduced_y_)) / len(self.y_)) * 100

        # Print reduction results
        # print("Results after performing ENN instance reduction technique:")
        # print("Number of instances in original data: " + str(self.X_.shape[0]))
        # print("Shape of original Y: "+str(self.y_.shape))
        # print("Number of instances in reduced data: " + str(self.reduced_X_.shape[0]))
        # print("Shape of reduced Y: "+str(self.reduced_y_.shape))
        # print("Reduction performed: " + str(round(self.reduction_, 4)) + " %")

        return self.reduced_X_, self.reduced_y_, self.reduction_


class ENN:
    """
    Edited Nearest Neighbors technique for instance reduction.
    Delete instances if they can be correctly classified without belonging to the training data. The misclassified ones
    will be kept in the reduced data.
    A classifier (k-NN model) must be given in order to perform instance reduction.
    """
    def __init__(self, classifier):
        self.y_ = None
        self.X_ = None
        self.reduced_X_ = None
        self.reduced_y_ = None
        self.reduction_ = None
        self.classes_ = None
        self.classifier = classifier

    def reduce_data(self, X, y):
        """
        Reduce data with ENN technique
        :param X: Features from training data (np.array-like of size nxm where n: num instances; m: num of features)
        :param y: Classes from training instances (np.array-like of size n where n: num instances)
        :return: reduced training data (X and y)
        """
        self.X_ = X
        self.y_ = y

        # Get the unique classes
        classes = np.unique(self.y_)
        self.classes_ = classes

        # Initialize indices arrays to perform reduction
        red_ind = np.ones(self.y_.size, dtype=bool)  # Final reduced data indices

        # Loop over all instances
        for i in range(y.size):
            # Simulate to delete instance from data
            red_ind[i] = False
            # Train k-NN model without deleted instance
            self.classifier.fit(self.X_[red_ind], self.y_[red_ind])
            # Save the deleted instance and class
            features, actual_class = self.X_[i], self.y_[i]
            # Check if predicting the deleted instance gives the actual class of it
            if self.classifier.predict(features.reshape(1, -1)) == [actual_class]:
                # If so, mark instance index for final deletion
                red_ind[i] = False
            else:
                red_ind[i] = True

        # Define reduced data
        self.reduced_X_ = np.asarray(X[red_ind])
        self.reduced_y_ = np.asarray(y[red_ind])

        # Check reduction (in terms of storage)
        self.reduction_ = (1.0 - float(len(self.reduced_y_)) / len(self.y_))*100

        # Print reduction results
        # print("Results after performing ENN instance reduction technique:")
        # print("Number of instances in original data: "+str(self.X_.shape[0]))
        # print("Shape of original Y: "+str(self.y_.shape))
        # print("Number of instances in reduced data: "+str(self.reduced_X_.shape[0]))
        # print("Shape of reduced Y: "+str(self.reduced_y_.shape))
        # print("Reduction performed: "+str(round(self.reduction_, 4))+" %")

        return self.reduced_X_, self.reduced_y_


class DROP2:
    """"
    Decremental Reduction Optimization Procedure technique for instance reduction.
    Delete instances from original data if the number of correctly classified instances without them in the training
    data is greater or equal to the number of classified instances with them in the training them.
    A classifiers (k-NN model) must be given in order to perform instance reduction.
    """

    def __init__(self, classifier, classifier_kplus1):
        self.reduced_X_ = None
        self.y_ = None
        self.X_ = None
        self.reduction_ = None
        self.reduced_y_ = None
        self.classifier = classifier
        self.classifier_kplus1 = classifier_kplus1

    def reduce_data(self, X, y):

        self.X_ = X
        self.y_ = y

        # Sort data by distance to the nearest "enemy"
        self.X_, self.y_ = self.sort_enemy_distance(self.X_, self.y_)

        # Initialize indices arrays to perform reduction
        mask = np.ones(self.y_.size, dtype=bool)
        red_ind = np.ones(self.y_.size, dtype=bool)

        # Loop over all instances
        for i in range(self.y_.size):
            # Fit the k+1-NN classifier with all instances (except for the ones already deleted)
            self.classifier_kplus1.fit(self.X_[red_ind], self.y_[red_ind])
            # Predict labels for all instances
            pred = self.classifier_kplus1.predict(self.X_)
            # Count how many predictions are correct (with P in S)
            correct_with = np.sum([self.y_ == pred])
            #print("Correct classifications with P in S: ", correct_with)

            # Simulate to delete instance from data (delete index)
            tmp_red_ind = red_ind.copy()
            tmp_red_ind[i] = False
            # Fit the k+1-NN classifier with all instances except for the deleted one
            self.classifier_kplus1.fit(self.X_[tmp_red_ind], self.y_[tmp_red_ind])
            # Predict labels for all instances (including the deleted one)
            pred = self.classifier_kplus1.predict(self.X_)
            # Count how many predictions are correct (without P in S)
            correct_without = np.sum([self.y_ == pred])
            #print("Correct classifications without P in S: ", correct_without)

            # Compare the number of correct predictions with and without P in S
            if correct_without >= correct_with:  # If without >= with, proceed to delete P
                red_ind[i] = False  # Define instance as to be deleted (dropped) (save index for deletion)
                # print(f"Deleting instance {i}...")

            # Change again the instance index to true (terminate deletion simulation)
            tmp_red_ind[i] = True

        # Define reduced data
        self.reduced_X_ = np.asarray(X[red_ind])
        self.reduced_y_ = np.asarray(y[red_ind])

        # Check reduction (in terms of storage)
        self.reduction_ = (1.0 - float(len(self.reduced_y_)) / len(self.y_)) * 100

        # Print reduction results
        # print("Results after performing DROP2 instance reduction technique: ")
        # print("Number of instances in original data: " + str(self.X_.shape[0]))
        # print("Shape of original Y: "+str(self.y_.shape))
        # print("Number of instances in reduced data: " + str(self.reduced_X_.shape[0]))
        # print("Shape of reduced Y: "+str(self.reduced_y_.shape))
        # print("Reduction performed: " + str(round(self.reduction_, 4)) + " %")

        return self.reduced_X_, self.reduced_y_

    def sort_enemy_distance(self, X, y):
        """
        Sort X and y by distance from each instance to its nearest enemy
        :param X: Features
        :param y: Class labels
        :return: Sorted features and class labels
        """
        # Initialize array to store indices
        ind_list = np.ones(y.size, dtype=bool)
        # Initialize list to store the distance to the nearest 'enemy' for each instance
        distance_list = []

        # Loop over all instances
        for i in range(y.size):
            # Create a bool list of indices from instances that are from a ≠ class than the actual instance ('enemies')
            ind_list = y != y[i]
            # Set the current instance index to 'False' to keep it out of 'training' data (to fit model)
            ind_list[i] = False
            # Fit the model with the selected instances
            self.classifier.fit(X[ind_list], y[ind_list])
            # Find distance to nearest 'enemy'. Reshape instance to pass it as array.
            n = self.classifier.find_kneighbors(X[i].reshape(1, -1))[0]
            # Append distance to distance_list
            distance_list.append(n)

        # Sort instances indexes by sorting distance_list from higher to lower
        sorted_list = sorted(range(len(distance_list)), key=lambda c: distance_list[c], reverse=True)

        # Define sorted data
        X_sorted = X[sorted_list]
        y_sorted = y[sorted_list]

        return X_sorted, y_sorted


class DROP3:
    """
    Decremental Reduction Optimization Procedure 3 technique for instance reduction.
    Perform instance reduction by applying both ENN and DROP2 techniques.
    """

    def __init__(self, classifier):
        self.classifier = classifier
        self.y_ = None
        self.X_ = None
        self.reduced_y_ = None
        self.reduced_X_ = None
        self.reduction_ = None
        self.classifier_kplus1 = KNN.KNeighborsClassifier(self.classifier.k + 1,
                                                          self.classifier.dist_metric,
                                                          self.classifier.voting_metric,
                                                          self.classifier.weights)

    def reduce_data(self, X, y):
        self.X_ = X
        self.y_ = y
        enn = ENN(self.classifier)
        X_enn, y_enn = enn.reduce_data(self.X_, self.y_)
        drop2 = DROP2(self.classifier, self.classifier_kplus1)
        self.reduced_X_, self.reduced_y_ = drop2.reduce_data(X_enn, y_enn)
        self.reduction_ = (1.0 - float(len(self.reduced_y_)) / len(self.y_))*100

        # Print reduction results
        # print("Results after performing DROP 3 instance reduction technique:")
        # print("Number of instances in original data: " + str(self.X_.shape[0]))
        # print("Shape of original Y: "+str(self.y_.shape))
        # print("Number of instances in reduced data: " + str(self.reduced_X_.shape[0]))
        # print("Shape of reduced Y: "+str(self.reduced_y_.shape))
        # print("Reduction performed: " + str(round(self.reduction_, 4)) + " %")

        return self.reduced_X_, self.reduced_y_, self.reduction_
