import util

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

train_label, raw_train_data = util.group_by_visit_number('data/train.csv')
test_label, raw_test_data = util.group_by_visit_number('data/test.csv', False)

# processed_train_data = util.process_data(raw_train_data, ['s', 'd'])
# processed_test_data = util.process_data(raw_test_data, ['s', 'd'])
#
# bag_of_features = util.get_feature_bag(processed_train_data, processed_test_data, {})
#
# train_data = util.one_hot_encoding(processed_train_data, bag_of_features)

group_trip = util.get_top_upc_by_trip_type(raw_train_data, train_label, top=30)
# print(group_trip)
# print(reduce(lambda x, y: x + y, map(lambda (key, value): len(value), group_trip.items())))
feature_bag = util.convert_to_upc_feature_bag(group_trip, {})
print(feature_bag)
print(len(feature_bag))
