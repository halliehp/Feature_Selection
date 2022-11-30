#  small dataset 46 & large dataset 78
import random
import pandas as pd
import math

df = pd.read_table('CS170_Small_Data__46.txt', engine='python', delimiter='  ',
                   names=['class label', 'feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature 5', 'feature 6'])
small_test_df = df.head(10)


#  leave one out cross validation stub code
def k_fold_cross_validation(data, current_set, feature_to_add):
    rand = round(random.random(), 3)
    return rand


def test_accuracy(data):
    count_of_correctly_classified = 0
    for i in range(len(data)):
        object_to_classify = data.iloc[i, 1:7]  # features
        print(object_to_classify)
        label_object_to_classify = data.iloc[i, 0]  # class label

        nearest_neighbor_distance = 0
        nearest_neighbor_location = 0
        for k in range(len(data)):
            if k is not i:
                print('Ask if', str(i+1), 'is nearest neighbor with', str(k+1))
                distance = math.sqrt(sum(object_to_classify-data.iloc[k, 1:7]))  # one feature or all of them ?
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data(nearest_neighbor_location, 0)
    print('Object', str(i), 'is class', str(label_object_to_classify))
    print('Its nearest neighbor is', str(nearest_neighbor_location), 'which is in class', str(nearest_neighbor_label))


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


"""print(df)
df1 = df.head(5)
print(df1)
feature_search(df1)"""
test_accuracy(small_test_df)

