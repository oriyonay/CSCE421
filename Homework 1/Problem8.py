# CSCE 421 HW #1 - Ori Yonay
# Libraries used: matplotlib, numpy, pandas, sklearn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# CONSTANTS:
DATASET_FILE = 'Smarket.csv'
K_TEST_RANGE = range(1, 11) # values of k to be tested

if __name__ == '__main__':
    # (1) Download and read the data:
    print('(1) Reading the data from CSV file ', DATASET_FILE, '...')
    data = pd.read_csv(DATASET_FILE)

    # (2) Print the data (first 5 lines):
    print('\n(2) Printing first 5 lines of the dataset:')
    print(data.head())

    # (3) Print the shape (dimensions) of the data:
    print('\n(3) Printing the shape of the data:')
    print(data.shape)

    # (4) Exract the features and the label from the data:
    print('\n(4) Extracting features and label data...')
    data_inputs = data[['Lag1', 'Lag2']]
    data_outputs = data[['Direction']]

    # convert data to numpy arrays:
    data_inputs = np.array(data_inputs)
    data_outputs = np.array(data_outputs).ravel()

    # (5) Split the data into a train/test split:
    print('\n(5) Splitting the data into train/test sets...')
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
        data_inputs, data_outputs, test_size = 0.2)

    # (6) Apply k-NN to the data:
    print('\n(6) Applying the k-NN to the dataset:')
    k_accuracy = [] # to log accuracy scores for k in [1, 10]

    for k in K_TEST_RANGE:
        print('\tTraining classifier on k = ', k)
        # create and train the classifier with k-neighbors:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(train_inputs, train_outputs)

        # predict the outputs using this classifier:
        predictions = knn.predict(test_inputs)

        # use predictions to calculate accuracy score:
        accuracy = accuracy_score(test_outputs, predictions)

        k_accuracy.append(accuracy)

    print('Training finished.')

    # (7) Plot the accuracy of your implementation for k in [1, 10]:
    print('\n(7) Plotting accuracy over k, saving into accuracy.png')
    fig = plt.plot(K_TEST_RANGE, k_accuracy)
    plt.xlabel = 'k'
    plt.ylabel = 'Accuracy'
    plt.savefig('accuracy.png')
    plt.close()

    print('\nDone.')
