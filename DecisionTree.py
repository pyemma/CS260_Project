import preprocess
import time

'''
TripType - a categorical id representing the type of shopping trip the customer made. This is the ground truth that you are predicting. TripType_999 is an "other" category.
VisitNumber - an id corresponding to a single trip by a single customer
Weekday - the weekday of the trip
Upc - the UPC number of the product purchased
ScanCount - the number of the given item that was purchased. A negative value indicates a product return.
DepartmentDescription - a high-level description of the item's department
FinelineNumber - a more refined category for each of the products, created by Walmart
'''
train_label, raw_train_data = preprocess.group_by_visit_number('data/train.csv')
test_label, raw_test_data = preprocess.group_by_visit_number('data/test.csv', False)

processed_train_data = preprocess.process_data(raw_train_data, ['s', 'd'])
processed_test_data = preprocess.process_data(raw_test_data, ['s', 'd'])

bag_of_features = preprocess.get_feature_bag(processed_train_data, processed_test_data, {})

train_data = preprocess.one_hot_encoding(processed_train_data, bag_of_features)
test_data = preprocess.one_hot_encoding(processed_test_data, bag_of_features)

print ("train_data len:", len(train_data), len(train_data[1]))
print ("test_data len:", len(test_data), len(test_data[1]))

from sklearn.tree import DecisionTreeClassifier
t1 = time.time()
clf = DecisionTreeClassifier()
clf = clf.fit(train_data, train_label)
t2 = time.time()
print ("training time:", (t2-t1))

t1 = time.time()
predicted = clf.predict(test_data)
t2 = time.time()
print ("prediction time:", (t2-t1))

print (predicted)
