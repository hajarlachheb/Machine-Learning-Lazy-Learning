import numpy as np


# Solve ties
def solve_ties_distances(tied_labels, k_labels, k_distances):
    label_distances = {}
    for i in range(len(k_labels)):
        label = k_labels[i]
        if label in tied_labels:
            label_distances[label] = label_distances.get(label, []) + [k_distances[i]]
    sorted_results = sorted(label_distances.items(), key=lambda item: np.mean(item[1]), reverse=False)
    return sorted_results[0][0]


# Majority class
def majority_class(k_labels, k_distances):
    unique_labels, label_counts = np.unique(k_labels, return_counts=True)
    max_freq = np.max(label_counts)
    if np.sum(label_counts == max_freq) > 1:
        tied_labels = unique_labels[label_counts == max_freq]
        return solve_ties_distances(tied_labels, k_labels, k_distances)
    else:
        return unique_labels[label_counts == max_freq]


# Inverse distance
def inverse_distance(k_labels, k_distances):
    w = 1 / np.array(k_distances, dtype='float64')
    return np.argmax(np.bincount(k_labels, weights=w))


# Sheppard's Work
def sheppards_work(k_labels, k_distances):
    w = np.exp(np.array(-k_distances, dtype='float64'))
    return np.argmax(np.bincount(k_labels, weights=w))
