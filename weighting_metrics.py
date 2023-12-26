from ReliefF import ReliefF
from sklearn.feature_selection import mutual_info_classif
import numpy as np


def information_gain(data, target):
    mi_score = mutual_info_classif(data, target, random_state=14)  # Set random_state for reproducible results
    information_gain_normalized = mi_score / np.sum(mi_score)
    return np.array(information_gain_normalized, dtype='object')


def relieff(data, target):
    fs = ReliefF(n_neighbors=100, n_features_to_keep=data.shape[1])
    fs.fit(data, target)
    reliefF_normalized = fs.feature_scores / np.sum(fs.feature_scores)
    return reliefF_normalized

