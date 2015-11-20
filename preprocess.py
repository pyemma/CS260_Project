import csv
import collections
"""process train data and test data"""


def process_data(filename=None, training=True):
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
