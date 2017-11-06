import os
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

K.set_image_dim_ordering('th')

with open(os.path.join(os.getcwd(), "config.json")) as fp:
    config = json.load(fp)

# Parameters ######################
# input image dimensions
img_rows, img_cols = config["img_rows"], config["img_cols"]

# number of channels
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
img_channels = config["img_channels"]

# Number of output classes
nb_classes = config["nb_classes"]

# Number of epochs to train
nb_epoch = config["nb_epoch"]

# Total number of convolutional filters to use
nb_filters = config["nb_filters"]
# Max pooling
nb_pool = config["nb_pool"]
# Size of convolution kernel
nb_conv = config["nb_conv"]
########################################

WeightFileName = config["WeightFileName"]
output = ["NOTHING", "SCISSOR", "ROCK", "PAPER"]


# This function does the guessing work based on input images
def guess_gesture(img):
    global output, get_output
    # Load image and flatten it
    image = np.array(img).flatten()

    # reshape it
    image = image.reshape(img_channels, img_rows, img_cols)

    # float32
    image = image.astype('float32')

    # normalize it
    image = image / 255

    # reshape for NN
    rimage = image.reshape(1, img_channels, img_rows, img_cols)

    # Now feed it to the NN, to fetch the predictions
    # index = model.predict_classes(rimage)
    # prob_array = model.predict_proba(rimage)

    prob_array = get_output([rimage, 0])[0]

    # print prob_array

    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1

    # Get the output with maximum probability
    import operator

    guess = max(d.items(), key=operator.itemgetter(1))[0]
    prob = d[guess]

    if prob > 70.0:
        print(guess + "  Probability: ", prob)

        return output.index(guess)

    else:
        return 1


# Load CNN model
def loadCNN(wf_index):
    global get_output
    model = Sequential()

    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                     padding='valid',
                     input_shape=(img_channels, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # Model summary
    model.summary()
    # Model conig details
    model.get_config()

    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)

    if wf_index >= 0:
        # Load pretrained weights
        fname = WeightFileName
        print("loading ", fname)
        model.load_weights(fname)

    layer = model.layers[11]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])

    return model
