#  small dataset 46 & large dataset 78
import pandas as pd
import math

df = pd.read_table('CS170_Small_Data__46.txt', engine='python', delimiter='  ',
                   names=['class label', '1', '2', '3', '4', '5', '6'])
test_df = pd.read_table('CS170_Small_Data__96.txt', engine='python', delimiter='  ',
                        names=['class label', '1', '2', '3', '4', '5', '6'])
small_test_df = df.head(20)


#  leave one out cross validation stub code
def k_fold_cross_validation(data, current_set, feature_to_add):  # returns accuracy
    # code to do k folding goes here
    features = ['class label']
    for x in current_set:
        features.append(str(x))
    features.append(str(feature_to_add))
    data = data[features]
    # print(data)
    # classification accuracy
    count_of_correctly_classified = 0
    for i in range(len(data)):
        object_to_classify = data.iloc[i, 1:]  # list of features at index i
        label_object_to_classify = data['class label'].iloc[i]  # class label at index i

        nearest_neighbor_distance = 10000
        nearest_neighbor_label = 0
        for k in range(len(data)):
            if k is not i:
                distance = math.sqrt(sum((object_to_classify - data.iloc[k, 1:]) ** 2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data['class label'].iloc[nearest_neighbor_location]
        if label_object_to_classify == nearest_neighbor_label and nearest_neighbor_label != 0:
            count_of_correctly_classified += 1
    accuracy = count_of_correctly_classified/len(data)
    # print('correctly classified out of 500:', count_of_correctly_classified)
    # print(accuracy)
    return accuracy


def feature_search(data):
    current_set_of_features = []
    best_accuracy = 0
    best_features = 0
    # forward selection
    for x in range(len(data.columns)-1):
        print('On the', x, 'th level of the search tree')
        best_so_far_accuracy = 0
        accuracy = 0

        for k in range(len(data.columns)-1):  # make sure to not count the class label column
            if k not in current_set_of_features:
                # current_set_of_features.append(k)
                accuracy = k_fold_cross_validation(data, current_set_of_features, k+1)
                print('---Using features', str(current_set_of_features), str(k), 'the accuracy is', accuracy)
            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_add_at_this_level = k
        current_set_of_features.append(feature_to_add_at_this_level)
        print('On level', str(x), 'I added feature', str(feature_to_add_at_this_level), 'to current set')
        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_features = current_set_of_features
    print('best accuracy:', best_accuracy)
    print('best features:', best_features)
    return best_features

# k_fold_cross_validation(test_df, [1, 3], 6)
feature_search(test_df)

