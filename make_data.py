import os

import json
from sklearn.model_selection import train_test_split


def get_data_path():
    current_path = os.path.abspath(os.curdir)
    data_path = os.path.join(current_path, '../data')

    return data_path


def write_annotation_file(file_path, X, y):
    with open(file_path, 'w') as f:
        for X_i, y_i in zip(X, y):
            f.write(X_i + ' ' + y_i + '\n')


def make_data(file_path):
    # json has this structure:
    # {image_path: label}
    # path from data folder
    with open(file_path, 'r') as f:
        data = json.load(f)

    paths = []
    labels = []
    for key in data.keys():
        paths.append(key)
        labels.append(data[key])

    X_train, X_val, y_train, y_val = train_test_split(paths, labels, test_size=0.1)
    write_annotation_file('data/train.txt', X_train, y_train)
    write_annotation_file('data/val.txt', X_val, y_val)

def get_real_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    paths = []
    labels = []
    for key in data.keys():
        if ('approved' in key) or ('json' in key):
            continue
        paths.append('cellcuts_test/' + key)

        label = data[key]
        if int(label) >= 1000:
            label = label[:-3] + ',' + label[-3:]
        labels.append(label)

    write_annotation_file('data/test.txt', paths, labels)

if __name__ == "__main__":
    data_path = get_data_path()
    # make_data(os.path.join(data_path, 'printed.json'))
    get_real_data(os.path.join(data_path, 'labels_test.json'))

# You have to change the source code in a few places:
#
# model.py, line 95: you should change the shape to (None, 1, 64, None) for img_data
# In cnn.py, you should add another max_2x1pool() layer. This is because the RNN expects a single-row output from
# the CNN, and the max-pooling makes the input height 32 times smaller by default.
# For an input of height 128, add another max_2x1pool() layer and change the shape to (None, 1, 128, None).
