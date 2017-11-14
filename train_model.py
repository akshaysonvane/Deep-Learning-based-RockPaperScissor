import os
import json
import time
import numpy as np
import matplotlib
import cnn_model as cnn

from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from keras.utils import np_utils


with open(os.path.join(os.getcwd(), "config.json")) as fp:
    config = json.load(fp)

batch_size = config["batch_size"]
nb_epoch = config["nb_epoch"]
nb_classes = config["nb_classes"]
img_channels = config["img_channels"]
img_rows = config["img_rows"]
img_cols = config["img_cols"]
data_set = config["data_set"]


def listdir(path):
    listing = os.listdir(path)
    ret_list = []
    for name in listing:
        # This check is to ignore any hidden files/folders
        if name.startswith('.'):
            continue
        ret_list.append(name)
    return ret_list


def train_model(model):
    # Split X and y into training and testing sets
    X_train, X_test, Y_train, Y_test = initialize()

    # Now start the training of the loaded model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                     verbose=1, validation_split=0.2)

    model.save_weights("new_weight_"+time.strftime("%Y%m%d-%H%M%S")+".hdf5")

    visualize_training_hist(hist)


def initialize():
    image_list = listdir(data_set)

    total_images = len(image_list)  # get the 'total' number of images

    # create matrix to store all flattened images
    image_matrix = np.array([np.array(Image.open(data_set + '/' + images).convert('L')).flatten()
                         for images in image_list], dtype='f')

    print(image_matrix.shape)

    input("Press any key")

    #########################################################
    ## Label the set of images per respective gesture type.
    ##
    label = np.ones((total_images,), dtype=int)

    samples_per_class = int(total_images / nb_classes)
    print("samples_per_class - ", samples_per_class)
    s = 0
    r = samples_per_class
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class

    '''
    # eg: For 301 img samples/gesture for 4 gesture types
    label[0:301]=0
    label[301:602]=1
    label[602:903]=2
    label[903:]=3
    '''

    data, Label = shuffle(image_matrix, label, random_state=2)
    train_data = [data, Label]

    (X, y) = (train_data[0], train_data[1])

    # Split X and y into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test


def visualize_training_hist(hist):
    # visualizing losses and accuracy

    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    xc = range(nb_epoch)

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    plt.savefig("train_loss vs val_loss")

    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    plt.savefig("train_acc vs val_acc")


def main():
    mod = cnn.loadCNN(-1)
    train_model(mod)


if __name__ == "__main__":
    main()
