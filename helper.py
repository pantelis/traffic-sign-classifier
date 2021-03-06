import pickle
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def load_pickled_data():
    training_file = '../traffic-signs-data/train.p'
    validation_file = '../traffic-signs-data/valid.p'
    testing_file = '../traffic-signs-data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def display_images_and_labels(images, labels, num_images):
    """Display the first image of each label."""
    if num_images == 1:
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        plt.title("Label {0}".format(labels))
        _ = plt.imshow(images, vmin=0, vmax=1)
        plt.show()
    else:
        assert (images.shape[0] == labels.shape[0])
        unique_labels = set(labels)
        plt.figure(figsize=(15, 15))
        i = 1
        for label in unique_labels:
            # Pick the first image for each label.
            image = images[labels.tolist().index(label)]
            plt.subplot(5, 10, i)  # A grid of 5 rows x 10 columns
            plt.axis('off')
            plt.title("Label {0} ({1})".format(label, labels.tolist().count(label)))
            i += 1
            _ = plt.imshow(image, vmin=0, vmax=1)
        plt.show()


def crop_image(image):
    ''''Crops an image and returns the cropped image.'''
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    # _ = plt.imshow(image)
    r = cv2.selectROI(image)
    # Cropped image
    cropped = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # plt.imshow(cropped)
    return cropped

def normalize(X_train, X_valid, X_test):
    '''Normalize image data'''
    X_train_norm = np.array(X_train / 255.0 - 0.5)
    X_valid_norm = np.array(X_valid / 255.0 - 0.5)
    X_test_norm = np.array(X_test / 255.0 - 0.5)

    return X_train_norm, X_valid_norm, X_test_norm
