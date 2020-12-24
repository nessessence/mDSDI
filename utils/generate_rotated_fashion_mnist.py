import pandas as pd
from scipy.ndimage import rotate as rot
import cv2
import numpy as np

def custom_rot(img, angle, reshape=False):
    _rot = rot(img, angle, reshape=reshape)
    mx, mn = 1, 0
    return (_rot - mn)/(mx - mn)

def load_fashion_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

X_train, y_train = load_fashion_mnist('/home/ubuntu/DSDI_data/fashion_MNIST', kind='train')
X_test, y_test = load_fashion_mnist('/home/ubuntu/DSDI_data/fashion_MNIST', kind='t10k')

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28 ,1)

X_train = X_train[:10000]
y_train = y_train[:10000]

tr_rotated_degree = [15, 30, 45, 60, 75]
for degree in tr_rotated_degree:
    train_paths, train_labels = [], []
    for idx in range(len(X_train)):
        train_image = X_train[idx]
        label = y_train[idx]
        train_path = str(degree) + "_degree/"+ str(label) + "/tr_image_" + str(idx) + ".png"
        train_paths.append(train_path)
        train_labels.append(label)

        train_image = custom_rot(train_image, angle=degree, reshape=False)
        cv2.imwrite("/home/ubuntu/DSDI_data/rotated_MNIST/fashion_MNIST/Raw images/"+ train_path, train_image)

    tr_meta_files = pd.DataFrame({'path': train_paths, 'label': train_labels})
    tr_meta_files.to_csv("/home/ubuntu/DSDI_data/rotated_MNIST/fashion_MNIST/Train val splits/"+ str(degree) + "_degree_train_kfold.txt", header=None, sep=' ', encoding='utf-8', index = False)

test_rotated_degree = [0, 15, 30, 45, 60, 75, 90]
for degree in test_rotated_degree:
    test_paths, test_labels = [], []
    for idx in range(len(X_test)):
        test_image = X_test[idx]
        label = y_test[idx]
        test_path = str(degree) + "_degree/"+ str(label) + "/test_image_" + str(idx) + ".png"
        test_paths.append(test_path)
        test_labels.append(label)

        test_image = custom_rot(test_image, angle=degree, reshape=False)
        cv2.imwrite("/home/ubuntu/DSDI_data/rotated_MNIST/fashion_MNIST/Raw images/"+ test_path, test_image)

    test_meta_files = pd.DataFrame({'path': test_paths, 'label': test_labels})
    test_meta_files.to_csv("/home/ubuntu/DSDI_data/rotated_MNIST/fashion_MNIST/Train val splits/"+ str(degree) + "_degree_test_kfold.txt", header=None, sep=' ', encoding='utf-8', index = False)