import preprocess

# """ compare the common value between train data and test data """
# fields = ['trip_type', 'visit_number', 'weekday',
#           'upc', 'scan_count', 'department_description', 'fine_line_number']
# train_dicts = preprocess.process_data('data/train.csv')
# test_dicts = preprocess.process_data('data/test.csv', False)
# index = 0
# for (dic1, dic2) in zip(train_dicts, test_dicts):
#     count = 0
#     for k1 in dic1.keys():
#         if k1 in dic2:
#             count = count + 1

#     print("Common values in " + fields[index] + ": " + str(count))
#     index = index + 1

train_label, raw_train_data = preprocess.group_by_visit_number('data/train.csv')
test_label, raw_test_data = preprocess.group_by_visit_number('data/test.csv', False)

processed_train_data = preprocess.process_data(raw_train_data, ['s', 'd'])
processed_test_data = preprocess.process_data(raw_test_data, ['s', 'd'])

bag_of_features = preprocess.get_feature_bag(processed_train_data, processed_test_data, {})

train_data = preprocess.one_hot_encoding(processed_train_data, bag_of_features)
