#  small dataset 46 & large dataset 78
import random
import pandas as pd
import math

df = pd.read_table('CS170_Small_Data__46.txt', engine='python', delimiter='  ',
                   names=['class label', '1', '2', '3', '4', '5', '6'])
test_df = pd.read_table('CS170_Small_Data__6.txt', engine='python', delimiter='  ',
                        names=['class label', '1', '2', '3', '4', '5', '6'])
small_test_df = df.head(20)


#  leave one out cross validation stub code
def k_fold_cross_validation(data, current_set, feature_to_add):  # returns accuracy
    # code to do k folding goes here
    features = []
    for x in current_set:
        features.append(str(x))
    features.append(str(feature_to_add))
    data_shaped = data[features]

    # classification accuracy
    count_of_correctly_classified = 0
    for i in range(len(data)):
        object_to_classify = data.iloc[i, 1:]  # list of features at index i
        # print(object_to_classify)
        label_object_to_classify = data['class label'].iloc[i]  # class label at index i

        nearest_neighbor_distance = 10000
        nearest_neighbor_location = 10000
        nearest_neighbor_label = 0
        for k in range(len(data.columns)-1):
            if k is not i:
                # print('Ask if', str(i), 'is nearest neighbor with', str(k))
                # print('object to classify', object_to_classify[k])
                # print('data shaped:', data.iloc[k])
                distance = math.sqrt(sum((object_to_classify-data.iloc[k, 1:])**2))  # one feature or all of them ?
                # print(distance)
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data['class label'].iloc[nearest_neighbor_location]
        # print('Object', str(i), 'is class', str(label_object_to_classify))
        # print('Its nearest neighbor is', str(nearest_neighbor_location), 'which is in class', str(nearest_neighbor_label))
        if label_object_to_classify == nearest_neighbor_label:
            count_of_correctly_classified += 1
    accuracy = count_of_correctly_classified/len(data)
    print(accuracy)
    return accuracy


def feature_search(data):
    current_set_of_features = []

    for x in range(len(data)):
        print('On the', x, 'th level of the search tree')
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0

        for k in range(len(data.columns)-1):  # make sure to not count the class label column
            if k not in current_set_of_features:
                current_set_of_features.append(k)
            print('---Considering adding the', str(k), 'feature')
            accuracy = k_fold_cross_validation(data, current_set_of_features, k+1)

            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_add_at_this_level = k
        print('On level', str(x), 'I added feature', str(feature_to_add_at_this_level), 'to current set')


k_fold_cross_validation(test_df, [2, 3], 5)
# print(small_test_df[['class label','2', '3']])

