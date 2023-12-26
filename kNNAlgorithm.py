# Import necessary libraries
import KNN

from load_datasets import load_vowel, load_adult, load_pen_based
from dist_metrics import minkowski_distance, cos_distance, HasD
from voting_metrics import inverse_distance, majority_class, sheppards_work
from weighting_metrics import information_gain, relieff
import numpy as np
import time


# Define function to run experiments
def run_experiments(x_train, y_train, x_test, y_test):
    """
    Run experiments and store the relevant information into a text file (.txt)
    :param x_train: Training data features
    :param y_train: Training data class labels
    :param x_test: Test data features
    :param y_test: Test data class labels
    """
    # Define lists to store relevant data
    acc_list = []
    time_list = []
    correct_pred_list = []
    incorrect_pred_list = []
    results_list = []
    k_res_list = []
    fold_list = []
    distance_list = []
    voting_list = []
    mean_acc_list = []
    weighting_metrics_list = []

    num_files = 10  # Number of files per each dataset

    # Define dictionaries and lists with all metrics
    distance_metrics_dic = {
        "Hassanat": HasD,
        "Minkowski": minkowski_distance,
        "Cosine": cos_distance
    }

    voting_metrics_dic = {
        "Majority Class": majority_class,
        "Inverse Distance": inverse_distance,
        "Sheppard's Work": sheppards_work
    }

    weighting_metrics_l = [
        "ReliefF",
        "Mutual Information Gain",
        "Equal"
    ]

    k_list = [1, 3, 5, 7]

    for dist_key in distance_metrics_dic:
        for k in k_list:
            for voting_key in voting_metrics_dic:
                for weighting_metric in weighting_metrics_l:
                    for i in range(0, 10):
                        if weighting_metric == "Mutual Information Gain":
                            weights = information_gain(x_train[i], y_train[i])
                        elif weighting_metric == "Equal":
                            weights = np.ones(x_train[i].shape[1])
                        elif weighting_metric == "ReliefF":
                            weights = relieff(x_train[i], y_train[i])

                        knn = KNN.KNeighborsClassifier(k=k, dist_metric=distance_metrics_dic[dist_key],
                                                       voting_metric=voting_metrics_dic[voting_key], weights=weights)

                        tmp_acc_list = []

                        start_time = time.time()
                        knn.fit(x_train[i], y_train[i])
                        acc, correct, incorrect = knn.evaluate(x_test[i], y_test[i])
                        end_time = time.time()

                        tmp_acc_list.append(acc)

                        # Store values into arrays
                        distance_list.append(dist_key)
                        voting_list.append(voting_key)
                        k_res_list.append(k)
                        weighting_metrics_list.append(weighting_metric)
                        acc_list.append(acc)
                        fold_list.append(i)
                        correct_pred_list.append(correct)
                        incorrect_pred_list.append(incorrect)
                        run_time = end_time - start_time
                        time_list.append(end_time - start_time)

                        results_list.append([dist_key, voting_key, k, i, acc, correct, incorrect, run_time])

                        print(f"Results for the fold number {i + 1}: ")

                        print(f"Accuracy score: {round(acc * 100, 2)}%")
                        print(f"Number of correctly predicted instances: {correct}")
                        print(f"Number of incorrectly predicted instances: {incorrect}")
                        print(f"Time to run: {run_time}")

                    mean_acc = np.mean(tmp_acc_list)
                    mean_acc_list.append(mean_acc * 100)
                    print(f"Mean accuracy for given model {round(mean_acc * 100, 2)}%")

    # Store data into a text file
    np.savetxt("AccuraciesResults.txt", mean_acc_list, fmt="%2.4f")  # Mean accuracies for all models

    # All data
    res = np.zeros(np.array(acc_list).size, dtype=[('distance', 'U16'), ('k', float), ('voting', 'U32'),
                                                   ('weighting', 'U32'), ('i', float),
                                                   ('acc', float), ('correct', float), ('incorrect', float),
                                                   ('run_time', float), ('mean_acc', float)])
    res['distance'] = distance_list
    res['k'] = k_res_list
    res['voting'] = voting_list
    res['weighting'] = weighting_metrics_list
    res['i'] = fold_list
    res['acc'] = acc_list
    res['correct'] = correct_pred_list
    res['incorrect'] = incorrect_pred_list
    res['run_time'] = time_list
    res['mean_acc'] = np.repeat(mean_acc_list, 10)
    np.savetxt("Results"+filename+".txt", res, fmt="%s %.0f %s %s %.0f %.4f %.4f %.4f %.8f %.5f")


# Define the file you want to apply instance reduction to

filename = str(input(f"Write the name of file to run k-NN experiments to (vowel, pen-based, adult): "))

if filename == 'pen-based':
    pen_based_x_train, pen_based_y_train, pen_based_x_test, pen_based_y_test = load_pen_based()
    print("Running experiments for the pen-based dataset...")
    run_experiments(pen_based_x_train, pen_based_y_train, pen_based_x_test, pen_based_y_test)
elif filename == 'vowel':
    vowel_x_train, vowel_y_train, vowel_x_test, vowel_y_test = load_vowel()
    print("Running experiments for the vowel dataset...")
    run_experiments(vowel_x_train, vowel_y_train, vowel_x_test, vowel_y_test)
elif filename == 'adult':
    adult_x_train, adult_y_train, adult_x_test, adult_y_test = load_adult()
    print("Running experiments for the adult dataset...")
    run_experiments(adult_x_train, adult_y_train, adult_x_test, adult_y_test)
else:
    raise ValueError('Unknown dataset {}'.format(filename))



