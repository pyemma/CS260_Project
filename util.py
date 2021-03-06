import csv
import collections
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
"""process train data and test data"""


def analyze_data(filename=None, training=True):
    if filename is None:
        raise ValueError("filename should not be None")

    trip_type = collections.Counter()
    visit_number = collections.Counter()
    weekday = collections.Counter()
    upc = collections.Counter()
    scan_count = collections.Counter()
    department_description = collections.Counter()
    fine_line_number = collections.Counter()

    dicts = []
    if training is True:
        dicts.append(trip_type)
    dicts.append(visit_number)
    dicts.append(weekday)
    dicts.append(upc)
    dicts.append(scan_count)
    dicts.append(department_description)
    dicts.append(fine_line_number)

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        reader.next()  # skip the header
        for row in reader:
            for i in range(0, len(row)):
                dicts[i][row[i]] = dicts[i][row[i]] + 1

        print("train data total number: " + str(reader.line_num))
        print("trip_type total number: " + str(len(trip_type)))
        print("visit_number total number: " + str(len(visit_number)))
        print("weekday total number: " + str(len(weekday)))
        print("upc total number: " + str(len(upc)))
        print("scan_count total number: " + str(len(scan_count)))
        print("department_description total number: " + str(len(department_description)))
        print("fine_line_number: " + str(len(fine_line_number)))

    if training is False:
        new_dicts = [{}]
        new_dicts.extend(dicts)
        dicts = new_dicts
    return dicts


def get_top_upc_or_fine_line_by_trip_type(
    data=None,
    label=None,
    top=20,
    f_type="upc"):
    """
    the data would be the result returned by group_by_visit_number,
    group the upc and its count within each trip_type,
    return top upc within each trip_type in the format of
    {'trip_type': [upc1, upc2, upc3, ...]}
    """
    assert(len(data) == len(label))
    group_trip_count = {}
    for i in range(1, len(data)):
        row = data[i]
        trip_type = label[i]
        if trip_type not in group_trip_count:
            group_trip_count[trip_type] = {}
        for (upc, count, desc, fine_line) in row[2]:
            if f_type == "upc":
                if upc not in group_trip_count[trip_type]:
                    group_trip_count[trip_type][upc] = 0
                group_trip_count[trip_type][upc] = group_trip_count[trip_type][upc] + 1
            elif f_type == "fine_line":
                if fine_line not in group_trip_count[trip_type]:
                    group_trip_count[trip_type][fine_line] = 0
                group_trip_count[trip_type][fine_line] = group_trip_count[trip_type][fine_line] + 1
            else:
                raise ValueError("not supported feature type")
    group_trip = {}
    for key in group_trip_count.keys():
        group_trip[key] = map(
            lambda (key, value): (value, key),
            group_trip_count[key].items())
        sorted(
            group_trip[key],
            key=lambda x: -x[0])
        group_trip[key] = group_trip[key][0:top]
        group_trip[key] = map(
        lambda (value, key): key, group_trip[key])
    return group_trip


def convert_to_feature_bag(group_trip, feature_bag={}):
    """
    take in output of get_top_upc_by_trip_type, and convert the data
    into feature bag with format
    {'feature': index}
    """
    feature_list = [elem for li in map(lambda (key, value): value, group_trip.items()) for elem in li]
    for elem in feature_list:
        if elem not in feature_bag:
            feature_bag[elem] = len(feature_bag)
    return feature_bag


def group_by_visit_number(filename=None, train=True):
    """
    data would be transformed to format like
    trip_type, (visit_number, weekday, [(upc, count, desc, fine_line)])
    """

    if filename is None:
        raise ValueError("Filename should not be none")

    aggregrated_data = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        reader.next()
        for row in reader:
            data = None
            if train is True:
                data = (row[0], (row[1], row[2]),
                        [(row[3], row[4], row[5], row[6])])
            else:
                data = (None, (row[0], row[1]),
                        [(row[2], row[3], row[4], row[5])])

            if data[1] in aggregrated_data:
                aggregrated_data[data[1]][2].append(data[2][0])
            else:
                aggregrated_data[data[1]] = data

    labels = []
    raw_data = []
    for (key, value) in aggregrated_data.items():
        label = value[0]
        data = (value[1][0], value[1][1], value[2])
        labels.append(label)
        raw_data.append(data)

    return (labels, raw_data)


def get_filter(fields_to_use=['u', 's', 'd', 'f'], reserve_visit_number=False):
    """ a helper function to generate a filter which will
        filter the raw data with the selected fiedls, this
        would only apply to the third term in the data representation
        return a lambda function which could apply to the raw data
        u: upc
        s: scan_count
        d: department_description
        f: fine_line_number """
    raw_fields = ['u', 's', 'd', 'f']
    mask_array = []
    for elem in raw_fields:
        if elem in fields_to_use:
            mask_array.append(True)
        else:
            mask_array.append(False)

    data_filter = None
    mask_array = np.array(mask_array)
    if reserve_visit_number is True:
        data_filter = lambda (x0, x1, x2): (x0, x1, map(lambda x: tuple(np.array(x)[mask_array]), x2))
    else:
        data_filter = lambda (x0, x1, x2): (x1, map(lambda x: tuple(np.array(x)[mask_array]), x2))

    return data_filter


def process_data(train_data=None,
                 fields_to_use=['u', 's', 'd', 'f'],
                 reserve_visit_number=False):
    """ filter data according to the fields to use """
    data_filter = get_filter(fields_to_use, reserve_visit_number)

    if train_data is None:
        raise ValueError("Train data should not be none")

    return map(data_filter, train_data)


def __update_bag(data=None, bag={}):

    dim = len(data[0])

    for row in data:
        for i in range(0, dim):
            if type(row[i]) is not list:
                if row[i] not in bag:
                    bag[row[i]] = len(bag)
            else:
                for elem in row[i]:
                    if elem not in bag:
                        bag[elem] = len(bag)

    return bag


def one_hot_encoding(data=None, bag={}, verbose=False, numerical=False):
    total_feature = len(bag)
    num_data = len(data)
    dim = len(data[0])
    sparse_data = np.zeros([num_data, total_feature])
    count = 0

    for j in range(0, num_data):
        row = data[j]
        for i in range(0, dim):
            if type(row[i]) is not list:
                if row[i] in bag:
                    index = bag[row[i]]
                    if numerical is False:
                        sparse_data[j][index] = 1
                    else:
                        sparse_data[j][index] = sparse_data[j][index] + 1
            else:
                for elem in row[i]:
                    if elem in bag:
                        index = bag[elem]
                        if numerical is False:
                            sparse_data[j][index] = 1
                        else:
                            sparse_data[j][index] = sparse_data[j][index] + 1
        count = count + 1
        if count % 1000 == 0 and verbose is True:
            print("Processed " + str(count))

    return sparse_data


def one_hot_encoding_upc_or_fine_line(
    data=None,
    bag={},
    f_type="upc",
    verbose=False):
    """
    one hot encode the upc data, different from one_hot_encode, the expact
    input data are expected to be the raw input instead of the prcoessed input.
    TODO: refactor the code in the furture
    """
    total_feature = len(bag)
    num_data = len(data)
    sparse_data = np.zeros([num_data, total_feature])
    hit_count = 0

    for j in range(0, num_data):
        row = data[j]
        for (upc, count, desc, fine_line) in row[2]:
            if f_type == "upc":
                if upc in bag:
                    hit_count = hit_count + 1
                    index = bag[upc]
                    sparse_data[j][index] = 1
            elif f_type == "fine_line":
                if fine_line in bag:
                    hit_count = hit_count + 1
                    index = bag[fine_line]
                    sparse_data[j][index] = 1
            else:
                raise ValueError("not supported feature type")

    print("Total hit count: " + str(hit_count))
    return sparse_data

def get_feature_bag(train_data=None,
                    test_data=None,
                    bag_of_features={}):
    """ one hot encoding of the data,
        retunr sparse matrix """

    assert(len(train_data[0]) == len(test_data[0]))

    bag_of_features = __update_bag(train_data, bag_of_features)
    bag_of_features = __update_bag(test_data, bag_of_features)
    total_feature = len(bag_of_features)
    print("Total feature number in bag: " + str(total_feature))

    return bag_of_features


def dump_result_to_file(filename='reuslt.csv', test_data=[], label=[], predict=[]):
    type_map = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, '12': 7, '14': 8,
                '15': 9, '18': 10, '19': 11, '20': 12, '21': 13, '22': 14, '23': 15, '24': 16, '25': 17,
                '26': 18, '27': 19, '28': 20, '29': 21, '30': 22, '31': 23, '32': 24, '33': 25, '34': 26,
                '35': 27, '36': 28, '37': 29, '38': 30, '39': 31, '40': 32, '41': 33, '42': 34, '43': 35, '44': 36, '999': 37}

    visit_number = [int(x0) for (x0, x1, x2) in test_data]
    visit_number_with_predict = zip(visit_number, predict)
    visit_number_with_predict.sort()

    result = np.zeros([len(visit_number_with_predict), 39])
    for i in range(0, len(visit_number_with_predict)):
        tup = visit_number_with_predict[i]
        result[i][0] = tup[0]
        index = type_map[tup[1]] + 1
        result[i][index] = 1

    np.savetxt('result.csv', result, fmt='%d', delimiter=',')
