import numpy as np
from keras.models import load_model
from keras import backend as K

from label_handler import LabelHandler
from aps_handler import APSHandler

APS_FOLDER = '/media/ben/Data/kaggle/passenger_screening_dataset/stage1/stage1_aps/'
TARGET_LABELS = '/media/ben/Data/kaggle/passenger_screening_dataset/stage1/stage1_sample_submission.csv'

label = LabelHandler(TARGET_LABELS)
subject_ids = label.get_subject_ids()
N = len(subject_ids)
print(N)

def predict(zone):
    model_path = '/media/ben/Data/kaggle/passenger_screening_dataset/stage1/models/epoch50_99_percent/{0}.h5'.format(zone)
    model = load_model(model_path)

    x_test = np.zeros((N, 16, 25, 25))

    i = 0
    for id_ in subject_ids:
        f = APS_FOLDER + id_ + '.aps'
        image = APSHandler(f)
        x = image.get_x(zone)
        x_test[i] = x
        i += 1

    img_rows, img_cols = 25, 25
    x_test = x_test.reshape(x_test.shape[0] * x_test.shape[1], img_rows, img_cols)
    print(x_test.shape)

    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    print(x_test.shape)

    y_test = model.predict(x_test, verbose=0)
    y_test = y_test[:,1]

    for i in range(x_test.shape[0]):
        # if zone in [2,4,7,10,12] and np.all(x_test[i] <= 0.1):
        if np.all(x_test[i] <= 0.1):
            y_test[i] = -1

    y_test = y_test.reshape(N, 16)
    y_test = np.average(y_test, weights=(y_test >= 0), axis=1)
    print(y_test)
    print(y_test.shape)
    print(zone, np.mean(y_test))

    return y_test

if __name__ == '__main__':
    import time
    start = time.time()
    with open('/media/ben/Data/kaggle/passenger_screening_dataset/stage1/result.csv', 'w') as result:
        result.write('Id,Probability\n')
        for zone in range(17):
            prob = predict(zone).tolist()
            for k in range(N):
                line = subject_ids[k] + '_Zone' + str(zone+1) + ',' + str(prob[k]) + '\n'
                result.write(line)

    end = time.time()
    lapsed_sec = end - start
    print(lapsed_sec)
