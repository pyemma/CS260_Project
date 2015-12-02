from sklearn.linear_model import LogisticRegression
import util
import numpy as np

"""
use logistic regression to classify data,
use top upc within each trip type as feature
"""

train_data_file_path = 'data/train.csv'
test_data_file_path = 'data/test.csv'

train_label, raw_train_data = util.group_by_visit_number(train_data_file_path)
test_label, raw_test_data = util.group_by_visit_number(test_data_file_path, False)

group_trip = util.get_top_upc_by_trip_type(
    data=raw_train_data,
    label=train_label,
    top=70)
bag_of_feature = util.convert_to_upc_feature_bag(
    group_trip=group_trip,
    feature_bag={})

print("Feature number: " + str(len(bag_of_feature)))

train_data = util.one_hot_encoding_upc(
    data=raw_train_data,
    bag=bag_of_feature,
    verbose=False)

count = 0
train_indexer = []
for i in range(0, len(train_data)):
    row = train_data[i]
    if sum(row) > 0:
        count = count + 1
        train_indexer.append(i)

print("Train data with true element: " + str(count))

test_data = util.one_hot_encoding_upc(
    data=raw_test_data,
    bag=bag_of_feature,
    verbose=False)

count = 0
test_indexer = []
for i in range(0, len(test_data)):
    row = test_data[i]
    if sum(row) > 0:
        count = count + 1
        test_indexer.append(i)

print("Test data with true element: " + str(count))

"""
use the mask to select out the data with feature,
use logistic regression to train the model
"""

train_label = np.array(train_label)
masked_train_data = train_data[np.array(train_indexer)]
masked_train_label = train_label[np.array(train_indexer)]
print(str(sum(masked_train_data[0])))

num_train_data = len(masked_train_data)
num_fold = 5
step_size = num_train_data / num_fold
result = []
for i in range(0, num_fold):
    start_index = i * step_size
    end_index = (i + 1) * step_size

    train_mask = np.ones(num_train_data, dtype=bool)
    test_mask = np.zeros(num_train_data, dtype=bool)
    train_mask[start_index:end_index] = np.zeros(step_size, dtype=bool)
    test_mask[start_index:end_index] = np.ones(step_size, dtype=bool)

    classifier = LogisticRegression()
    classifier.fit(masked_train_data[train_mask], masked_train_label[train_mask])

    """
    training error
    """
    predicts = classifier.predict(masked_train_data[train_mask])
    correct = 0
    error = 0
    for tup in zip(list(predicts), list(masked_train_label[train_mask])):
        if tup[0] == tup[1]:
            correct = correct + 1
        else:
            error = error + 1
    print("Training accu: " + str(float(correct) / float(correct + error)))

    predicts = classifier.predict(masked_train_data[test_mask])

    correct = 0
    error = 0
    for tup in zip(list(predicts), list(masked_train_label[test_mask])):
        if tup[0] == tup[1]:
            correct = correct + 1
        else:
            error = error + 1
    result.append(float(correct) / float(correct + error))

print(str(result))
cross_avg_accu = sum(result) / float(num_fold)
print(
    "Cross validation result for upc feature is " +
    str(float(cross_avg_accu)))
