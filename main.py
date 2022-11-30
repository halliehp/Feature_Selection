#  small dataset 46 & large dataset 78
import random
import pandas as pd

df = pd.read_table('CS170_Small_Data__46.txt', delimiter='  ',
                   names=['class label', 'feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature 5', 'feature 6'])


#  leave one out cross validation stub code
def k_fold_cross_validation(data, current_set, feature_to_add):
    rand = round(random.random(), 3)
    return rand


def feature_search(data):
    current_set_of_features = []

    for x in range(len(data)):
        print('On the ', x, 'th level of the search tree')
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0

        for k in range(len(data.columns)-1):  # make sure to not count the class label column
            if k not in current_set_of_features:
                current_set_of_features.append(k)
            print('---Considering adding the ', str(k), ' feature')
            accuracy = k_fold_cross_validation(data, current_set_of_features, k+1)

            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_add_at_this_level = k
        print('On level ', str(x), 'I added feature ', str(feature_to_add_at_this_level), ' to current set')


print(df)
df1 = df.head(5)
print(df1)
feature_search(df1)

