import time
from load_datasets import load_vowel, load_adult, load_pen_based
import KNN
from dist_metrics import minkowski_distance, cos_distance
from voting_metrics import inverse_distance, sheppards_work
from weighting_metrics import relieff
import numpy as np
from ReductionTechniques import RENN, DROP3, RNN

# Choose the instance reduction to perform:
option = int(input("Choose the number of the Instance Reduction option you'd like to run:"
                   "\n 1: RNN (Reduced Nearest Neighbor)"
                   "\n 2: RENN (Repeated Edited Nearest Neighbor)"
                   "\n 3: DROP3 (Decremental Reduction Optimization Procedure 3)"
                   "\nWrite the desired option (number): "))

# Define the file you want to apply instance reduction to
filename = str(input(f"Write the name of file to run instance reduction to (vowel, pen-based, adult): "))

if filename == 'pen-based':
    xt, yt, xtt, ytt = load_pen_based()
    print("Performing instance reduction on the pen-based dataset...")
elif filename == 'vowel':
    xt, yt, xtt, ytt = load_vowel()
    print("Performing instance reduction on the vowel dataset...")
elif filename == 'adult':
    xt, yt, xtt, ytt = load_adult()
    print("Performing instance reduction on the adult dataset...")
else:
    raise ValueError('Unknown dataset {}'.format(filename))

# Define lists
acc_list = []
time_list = []
reduction_list = []
i_list = []
reduction_technique_l = []

# Loop over all folds
for i in range(0, 10):

    if filename == 'adult':
        # Reduce number of instances to 1000 for adult dataset
        xt[i] = xt[i][:1000, :]
        yt[i] = yt[i][:1000]
        xtt[i] = xtt[i]
        ytt[i] = ytt[i]
        # Define best classifier for adult dataset
        weights = np.ones(xt[i].shape[1])  # Define equal weights
        knn = KNN.KNeighborsClassifier(k=3, dist_metric=cos_distance, voting_metric=inverse_distance, weights=weights)

    elif filename == 'vowel':
        # Define best classifier for vowel dataset
        weights = relieff(xt[i], yt[i])  # Define reliefF weights
        knn = KNN.KNeighborsClassifier(k=1, dist_metric=minkowski_distance, voting_metric=sheppards_work,
                                       weights=weights)

    elif filename == 'pen-based':
        # Reduce number of instances to 1000 for adult dataset
        # xt[i] = xt[i][:1000, :]
        # yt[i] = yt[i][:1000]
        # xtt[i] = xtt[i]
        # ytt[i] = ytt[i]
        # Define the best classifier for pen-based dataset
        weights = np.ones(xt[i].shape[1])  # Define equal weights
        knn = KNN.KNeighborsClassifier(k=1, dist_metric=minkowski_distance, voting_metric=sheppards_work,
                                       weights=weights)

    # Define the reduction technique depending on the option chosen by the user
    if option == 1:
        reduction_technique = 'RNN'
        tec = RNN(knn)
    elif option == 2:
        reduction_technique = 'RENN'
        tec = RENN(knn, 5)
    elif option == 3:
        reduction_technique = 'DROP3'
        tec = DROP3(knn)
    else:
        raise ValueError("There is no option {}".format(option))

    # Get reduced data and reduction percentage
    X_reduced, y_reduced, red_percentage = tec.reduce_data(xt[i], yt[i])

    X_train = X_reduced
    Y_train = y_reduced
    X_test = xtt[i]
    Y_test = ytt[i]

    start = time.time()  # Starting time
    knn.fit(X_train, Y_train)  # Fit k-NN model
    acc, correct, incorrect = knn.evaluate(X_test, Y_test)  # Predict and obtain evaluation results
    end = time.time()  # End time

    time_list.append(end-start)
    acc_list.append(acc*100)

    reduction_list.append(red_percentage)

    i_list.append(i)

    reduction_technique_l.append(reduction_technique)

# All data
res = np.zeros(np.array(acc_list).size, dtype=[('technique', 'U32'), ('i', float), ('accuracy', float), ('time', float),
                                               ('reduction', float)])
res['accuracy'] = acc_list
res['time'] = time_list
res['reduction'] = reduction_list
res['technique'] = reduction_technique_l
res['i'] = i_list

np.savetxt(reduction_technique+"_results_"+filename+".txt", res, fmt="%s %.0f %2.4f %.8f %.5f")

mean_acc = np.mean(np.array(acc_list))
mean_time = np.mean(np.array(time_list))
mean_reduction = np.mean(np.array(reduction_list))

print(f"Results after performing reduction with {reduction_technique} on the 10 folds of the {filename} dataset:")
print("The mean accuracy is: "+str(round(mean_acc, 5))+" %")
print("The mean time to run k-NN is: "+str(round(mean_time, 6))+" seconds")
print("The percentage of reduction has been: "+str(round(mean_reduction, 4))+" %")



