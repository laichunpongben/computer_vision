'''Trains a simple convnet on the passenger_screening dataset.
'''

from __future__ import print_function
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split

def train(zone):
    batch_size = 128
    num_classes = 2
    epochs = 100

    img_rows, img_cols = 25, 25

    fpath = '/media/ben/Data/kaggle/passenger_screening_dataset/stage1/'
    npz_path = fpath + str(zone) + '.npz'
    d = np.load(npz_path)
    x, y = d['x'], d['y']
    x = x.reshape(x.shape[0] * x.shape[1], img_rows, img_cols)
    y = y.repeat(16)
    print(x.shape)
    print(y.shape)

    to_remove = []
    threshold = 0.1
    for i in range(x.shape[0]):
        if np.all(x[i] <= threshold):
            to_remove.append(i)

    # print(to_remove)
    x = np.delete(x, to_remove, 0)
    y = np.delete(y, to_remove, 0)
    print(x.shape)
    print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    model.save('/media/ben/Data/kaggle/passenger_screening_dataset/stage1/{0}.h5'.format(zone))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    import time
    start = time.time()
    for zone in range(17):
        train(zone)
    end = time.time()
    lapsed_sec = end - start
    print(lapsed_sec)
