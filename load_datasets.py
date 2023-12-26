import os
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_pen_based():
    # Define each one of the train and test files obtained from the 10-fold cross validation:
    test_files = (
        'pen-based.fold.000000.test.arff', 'pen-based.fold.000001.test.arff', 'pen-based.fold.000002.test.arff',
        'pen-based.fold.000003.test.arff', 'pen-based.fold.000004.test.arff', 'pen-based.fold.000005.test.arff',
        'pen-based.fold.000006.test.arff', 'pen-based.fold.000007.test.arff', 'pen-based.fold.000008.test.arff',
        'pen-based.fold.000009.test.arff')
    train_files = (
        'pen-based.fold.000000.train.arff', 'pen-based.fold.000001.train.arff', 'pen-based.fold.000002.train.arff',
        'pen-based.fold.000003.train.arff', 'pen-based.fold.000004.train.arff', 'pen-based.fold.000005.train.arff',
        'pen-based.fold.000006.train.arff', 'pen-based.fold.000007.train.arff', 'pen-based.fold.000008.train.arff',
        'pen-based.fold.000009.train.arff')

    # Define new arrays that will contain the preprocessing result for training and test data sets
    x_test = []
    y_test = []
    x_train = []
    y_train = []

    # Preprocessing
    # Test files - iterate over all files in the tuple defined above
    for testfile in test_files:
        testds = os.path.join('datasetsCBR/pen-based', testfile)
        data = loadarff(testds)
        df_data = pd.DataFrame(data[0])

        y_test.append(
            df_data['a17'].astype(int).to_numpy())  # Store classes in corresponding array with numerical format
        df_data.drop('a17', axis=1, inplace=True)  # Drop classes from dataframe

        scaler = MinMaxScaler()
        test_array = scaler.fit_transform(df_data)  # Scale all features to a [0, 1] range w/ MinMaxScaler
        x_test.append(test_array)  # Append results to the final array

    for trainfile in train_files:
        trainds = os.path.join('datasetsCBR/pen-based', trainfile)
        data = loadarff(trainds)
        df_data = pd.DataFrame(data[0])

        y_train.append(
            df_data['a17'].astype(int).to_numpy())  # Store classes in corresponding array with numerical format
        df_data.drop('a17', axis=1, inplace=True)  # Drop classes from dataframe

        scaler = MinMaxScaler()
        train_array = scaler.fit_transform(df_data)  # Scale all features to a [0, 1] range w/ MinMaxScaler
        x_train.append(train_array)  # Append results to the final array

    return np.array(x_train, dtype='object'), np.array(y_train, dtype='object'), np.array(x_test, dtype='object'), \
           np.array(y_test, dtype='object')


def load_vowel():
    # Define each one of the train and test files obtained from the 10-fold cross validation:
    test_files = ('vowel.fold.000000.test.arff', 'vowel.fold.000001.test.arff', 'vowel.fold.000002.test.arff',
                  'vowel.fold.000003.test.arff', 'vowel.fold.000004.test.arff', 'vowel.fold.000005.test.arff',
                  'vowel.fold.000006.test.arff', 'vowel.fold.000007.test.arff', 'vowel.fold.000008.test.arff',
                  'vowel.fold.000009.test.arff')
    train_files = ('vowel.fold.000000.train.arff', 'vowel.fold.000001.train.arff', 'vowel.fold.000002.train.arff',
                   'vowel.fold.000003.train.arff', 'vowel.fold.000004.train.arff', 'vowel.fold.000005.train.arff',
                   'vowel.fold.000006.train.arff', 'vowel.fold.000007.train.arff', 'vowel.fold.000008.train.arff',
                   'vowel.fold.000009.train.arff')

    # Define new arrays that will contain the preprocessing result for training and test data sets
    x_test = []
    y_test = []
    x_train = []
    y_train = []

    # Preprocessing
    # Test files - iterate over all files in the tuple defined above
    for testfile in test_files:
        testds = os.path.join('datasetsCBR/vowel', testfile)
        data = loadarff(testds)  # Load arff data
        df_data = pd.DataFrame(data[0])  # Transform loaded data into a pandas DF

        test_array, test_y = preprocessVowel(df_data)  # Preprocess data
        x_test.append(test_array)  # Append results to the final array
        y_test.append(test_y)  # Append results to the outputs array

    # Train files - iterate over all files in the tuple defined above
    for trainfile in train_files:
        trainds = os.path.join('datasetsCBR/vowel', trainfile)
        data = loadarff(trainds)  # Load arff data
        df_data = pd.DataFrame(data[0])  # Transform loaded data into a pandas DF

        train_array, train_y = preprocessVowel(df_data)  # Preprocess data
        x_train.append(train_array)  # Append results to the final array
        y_train.append(train_y)  # Append results to the outputs array

    return np.array(x_train, dtype='object'), np.array(y_train, dtype='int'), np.array(x_test, dtype='object'), \
           np.array(y_test, dtype='int')


def load_adult():
    test_files = ('adult.fold.000000.test.arff', 'adult.fold.000001.test.arff', 'adult.fold.000002.test.arff',
                  'adult.fold.000003.test.arff', 'adult.fold.000004.test.arff', 'adult.fold.000005.test.arff',
                  'adult.fold.000006.test.arff', 'adult.fold.000007.test.arff', 'adult.fold.000008.test.arff',
                  'adult.fold.000009.test.arff')
    train_files = ('adult.fold.000000.train.arff', 'adult.fold.000001.train.arff', 'adult.fold.000002.train.arff',
                   'adult.fold.000003.train.arff', 'adult.fold.000004.train.arff', 'adult.fold.000005.train.arff',
                   'adult.fold.000006.train.arff', 'adult.fold.000007.train.arff', 'adult.fold.000008.train.arff',
                   'adult.fold.000009.train.arff')

    # Define new arrays that will contain the preprocessing result for training and test data sets
    x_test = []
    y_test = []
    x_train = []
    y_train = []

    # Preprocessing
    # Test files - iterate over all files in the tuple defined above
    for testfile in test_files:
        dataset = os.path.join('datasetsCBR/adult', testfile)
        data = loadarff(dataset)
        df_data = pd.DataFrame(data[0])

        test_array, test_y = preprocessAdult(df_data)  # Preprocess data
        x_test.append(test_array)  # Append results to final array
        y_test.append(test_y)  # Append results to the outputs array

    # Train data
    for trainfile in train_files:
        dataset = os.path.join('datasetsCBR/adult', trainfile)
        data = loadarff(dataset)
        df_data = pd.DataFrame(data[0])

        train_array, train_y = preprocessAdult(df_data)  # Preprocess data
        x_train.append(train_array)  # Append results to final array
        y_train.append(train_y)  # Append results to the outputs array

    return np.array(x_train, dtype='object'), np.array(y_train, dtype='object'), np.array(x_test, dtype='object'), \
           np.array(y_test, dtype='object')


def preprocessAdult(df_data):
    df_data.rename(columns={'capital-gain': 'capital gain', 'capital-loss': 'capital loss', 'native-country': 'country',
                            'hours-per-week': 'hours per week', 'marital-status': 'marital'}, inplace=True)

    # Replace '?' symbols for NaN values
    df_data['country'] = df_data['country'].replace('?', np.nan)
    df_data['workclass'] = df_data['workclass'].replace('?', np.nan)
    df_data['occupation'] = df_data['occupation'].replace('?', np.nan)

    df_data.dropna(how='any', inplace=True)  # Drop missing values

    # Encode the dataset with LabelEncoder
    le = LabelEncoder()
    cols = list(df_data.select_dtypes(include=['object']).columns)
    df_data[cols] = df_data[cols].apply(le.fit_transform)  # Encode categorical var (cols features) w/ LabelEncoder

    classes = df_data['class'].to_numpy()  # Store the classes in a variable
    df_data.drop('class', axis=1, inplace=True)  # Drop classes from dataframe

    # Scale the data with MinMaxScaler
    min_max_scaler = MinMaxScaler()
    array = min_max_scaler.fit_transform(df_data)  # Scale all features to a [0, 1] range w/ MinMaxScaler

    return array, classes


def preprocessVowel(df_data):
    cols = ['Sex', 'Train_or_Test', 'Speaker_Number', 'Class']
    le = LabelEncoder()
    df_data[cols] = df_data[cols].apply(le.fit_transform)  # encode categorical var (cols features) w/ LabelEncoder

    classes = df_data['Class'].to_numpy()
    df_data.drop('Class', axis=1, inplace=True)

    scaler = MinMaxScaler()
    array = scaler.fit_transform(df_data)  # Scale all features to a [0, 1] range w/ MinMaxScaler

    return array, classes

