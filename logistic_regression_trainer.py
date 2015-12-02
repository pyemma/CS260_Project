from sklearn.linear_model import LogisticRegression
import util
import numpy as np


""" use logistic regression to classify data,
    this method does not support huge number of feature """

# change to your data file path
train_data_file_path = 'data/train.csv'
test_data_file_path = 'data/test.csv'

train_lable, raw_train_data = util.group_by_visit_number(train_data_file_path)
test_label, raw_test_data = util.group_by_visit_number(test_data_file_path, False)

# feature to be tested with
feature_set = [['d'], ['s', 'd'], ['f'], ['s', 'f']]
feature_result = []

for feature in feature_set:
    # filter the data with the required feature
    pro_train_data = util.process_data(raw_train_data, feature)
    pro_test_data = util.process_data(raw_test_data, feature)

    bag_of_features = util.get_feature_bag(pro_train_data, pro_test_data, {})
    train_data = util.one_hot_encoding(pro_train_data, bag_of_features)
    train_lable = np.array(train_lable)

    # 5-fold cross validation
    num_train_data = len(train_data)
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
        classifier.fit(train_data[train_mask], train_lable[train_mask])

        predicts = classifier.predict(train_data[test_mask])

        correct = 0
        error = 0
        for tup in zip(list(predicts), list(train_lable[test_mask])):
            if tup[0] == tup[1]:
                correct = correct + 1
            else:
                error = error + 1
        result.append(float(correct) / float(error + correct))

    cross_avg_accu = sum(result) / float(num_fold)
    print(
        "Cross validation result for feature " +
        str(feature) + " is " + str(float(cross_avg_accu)))
    feature_result.append(cross_avg_accu)

print(str(zip(feature_set, feature_result)))
