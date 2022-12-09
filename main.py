#  small dataset 46 & large dataset 78
import pandas as pd
import numpy as np
import copy
import time

df = pd.read_table('CS170_Small_Data__46.txt', engine='python', delimiter='  ',
                   names=['class label', '1', '2', '3', '4', '5', '6'])
large_df = pd.read_table('CS170_Large_Data__78.txt', engine='python', delimiter='  ',
                         names=['class label', '1', '2', '3', '4', '5', '6'])
test_df = pd.read_table('CS170_Small_Data__88.txt', engine='python', delimiter='  ',
                        names=['class label', '1', '2', '3', '4', '5', '6'])
small_test_df = df.head(20)
arr = test_df.to_numpy()


def calculate_distance(a, b):
    sub = a - b
    return np.dot(sub, sub)


def k_fold_cross_validation_anti_pandas(data, current_set, feature_to_add):  # returns accuracy
    # code to do k folding goes here
    features = copy.deepcopy(current_set)
    if feature_to_add != 0:
        features.append(feature_to_add)

    count_of_correctly_classified = 0
    for i in range(len(data)):
        object_to_classify = []
        for p in range(len(features)):
            object_to_classify.append(data[i][features[p]])  # list of features at index i
        label_object_to_classify = data[i][0]  # class label at index i

        nearest_neighbor_distance = np.inf
        nearest_neighbor_label = 0
        for k in range(len(data)):
            if k != i:
                temp = []
                for s in range(len(features)):
                    temp.append(data[k][features[s]])  # list of features at index k
                temp = np.array(temp)
                distance = calculate_distance(object_to_classify, temp)
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location][0]
        if label_object_to_classify == nearest_neighbor_label:
            count_of_correctly_classified += 1
    accuracy = count_of_correctly_classified/len(data)
    accuracy = round(accuracy, 3)
    # print('correctly classified out of 500:', count_of_correctly_classified)
    # print(accuracy)
    return accuracy


def feature_search(file_name):
    data = pd.read_table(file_name, engine='python', delimiter='  ')
    current_set_of_features = []
    best_accuracy = 0
    best_features = 0
    # forward selection
    for x in range(len(data.columns)-1):
        print('On the', x, 'th level of the search tree')
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0

        for k in range(len(data.columns)-1):  # make sure to not count the class label column
            if k+1 not in current_set_of_features:
                pls = data.to_numpy()
                accuracy = k_fold_cross_validation_anti_pandas(pls, current_set_of_features, k+1)
                print('---Using features', str(current_set_of_features), str(k+1), 'the accuracy is', accuracy)
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k+1
        print('On level', str(x), 'I added feature', str(feature_to_add_at_this_level), 'to current set')
        current_set_of_features.append(feature_to_add_at_this_level)
        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_features = copy.deepcopy(current_set_of_features)
    print('\nSearch finished!')
    print('best accuracy:', best_accuracy)
    print('using these features:', best_features)
    return best_features


def backwards_elimination(file_name):
    data = pd.read_table(file_name, engine='python', delimiter='  ')
    current_set_of_features = []
    best_accuracy = 0
    best_features = 0
    pls = data.to_numpy()
    for x in range(len(data.columns) - 1):
        current_set_of_features.append(x+1)
    for x in range(len(data.columns)-1):
        print('On the', x, 'th level of the search tree')
        feature_to_remove = []
        best_so_far_accuracy = k_fold_cross_validation_anti_pandas(pls, current_set_of_features, 0)
        for k in range(len(data.columns)-1):
            exists = current_set_of_features.count(k+1)
            if exists > 0:
                elimination_test = copy.deepcopy(current_set_of_features)
                elimination_test.remove(k+1)
                accuracy = k_fold_cross_validation_anti_pandas(pls, elimination_test, 0)
                print('---Using features', str(elimination_test), 'the accuracy is', accuracy)
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_remove = k+1
        try:
            current_set_of_features.remove(feature_to_remove)
            print('On level', str(x), 'I removed feature', str(feature_to_remove), 'from current set')
        except:
            print('List is empty')
        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_features = copy.deepcopy(current_set_of_features)
    print('\nSearch finished!')
    print('best accuracy:', best_accuracy)
    print('using these features:', best_features)
    return best_features


def menu():
    print('**Feature Selection and Nearest Neighbor Classification Algorithm**\n')
    file_name = input('Type in the name of the file to test: ')
    print('\n(1) Forward Selection\n(2) Backwards Elimination')
    algorithm = input('Type the number of the algorithm you want to run: ')
    if algorithm == '1':
        print('\n Starting Forward Selection')
        start = time.time()
        feature_search(file_name)
        end = time.time()
        elapsed = round(end - start, 1)
        print('time elapsed: ', elapsed)
    if algorithm == '2':
        print('\n Starting Backwards Elimination')
        start = time.time()
        backwards_elimination(file_name)
        end = time.time()
        elapsed = round(end - start, 1)
        print('time elapsed: ', elapsed)


menu()

