import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import time
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

import cnn_enemy_detector as detect


def plot_img(imgs, label, start, end):
    """

    :param imgs: images
    :param label: ground truth label
    :param start: print starting from img index
    :param end: print until img index
    :return:
    """

    if end - start <= 10:
        for i in range(start, end):

            plt.figure(i-start)
            if color:
                img = imgs[i]
            else:
                img = imgs[i][:, :, 0]
            plt.imshow(img)
            plt.title(str(label[i]))

        plt.show()
    else:
        print('you tried to plot {} images'.format(end - start))


def compress_image(img_path, size, color):
    """

    :param img_path: an image
    :param size: new size for CNN input
    :param color: if color img wanted
    :return: 3D tensor of gray img (w, h, 1) or
             3D tensor of color img (w, h, 3)
    """

    img = cv2.imread(img_path)
    if color:
        return np.expand_dims(img, axis=2)
    else:
        # get gray img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # normalize
        #img = img / 255.0
        # resize
        img = cv2.resize(img, size)
        # for gray img increase dim by 1 as input of conv2d requires 4 dim
        img = np.expand_dims(img, axis=2)

        return img


def get_compressed_imgs(img_paths, data_type, size, color=False):
    """

    :param img_paths: image paths
    :param data_type: train, val, test data
    :param size: input size for CNN
    :param color: set True, if color images wanted
    :return: 4D tensor (N, w, h, 1) or (N, w, h, 3)
    """

    compressed_img = []
    for img in img_paths:
        compressed_img.append(compress_image(img, size, color=color))

    # normalize by subtracting the mean channel values
    compressed_img = np.array(compressed_img)
    mean_yuv = np.mean(compressed_img, axis=(0, 1, 2))

    print('{} data mean value = {}'.format(data_type, mean_yuv))
    compressed_img = compressed_img - mean_yuv

    return compressed_img


def load_data(path, robot):
    """

    :param path: img paths
    :param robot: if from robot folder, else not robot
    :return: list of img paths and corresponding labels
    """
    imgs_list = []

    for file in glob.glob(path + '*.png'):
        imgs_list.append(file)

    num_img = len(imgs_list)
    if robot:
        label = np.ones([num_img])
    else:
        label = np.zeros([num_img])

    return imgs_list, list(label)


def get_data_batch(img_r, img_nr, l1, l2, batch_num=5, show_data=False):
    """

    :param img_r: robot images
    :param img_nr: not robot images
    :param l1: label of robot images
    :param l2: label of not robot images
    :param batch_num: num of batches
    :param show_data: print num of data
    :return: train, val, test sets
    """

    # create combined list
    data = img_r + img_nr
    label = l1 + l2
    # split into train and test
    X_train1, X_test, y_train1, y_test = train_test_split(data, label, test_size=0.15, random_state=42)
    # split into train and validate
    X_train2, X_val, y_train2, y_val = train_test_split(X_train1, y_train1, test_size=0.17, random_state=42)

    # batch size of training batch
    batch_size = math.floor(len(X_train2)/batch_num)
    # additional trainings data add to validation set
    left_over = len(X_train2) - batch_size*batch_num

    if left_over > 0:
        # append left over data to validation set
        X_val = X_val + X_train2[-left_over:]
        y_val = y_val + y_train2[-left_over:]
        #print('val_data = {}'.format(len(X_val)))

    X_train = []
    y_train = []
    s = 0
    e = batch_size
    for i in range(batch_num):
        X_train.append(X_train2[s:e])
        y_train.append(y_train2[s:e])
        s = s + batch_size
        e = e + batch_size

    data_train = [X_train, y_train]
    data_val = [X_val, y_val]
    data_test = [X_test, y_test]

    if show_data:
        print('num of train data = {}'.format(len(X_train2)))
        print('num of train data batch = {}'.format(len(X_train[0])))
        print('num of val data = {}'.format(len(X_val)))
        print('num of test data = {}'.format(len(X_test)))

    return data_train, data_val, data_test


if __name__=='__main__':

    # get time
    start_time = time.time()

    """ pre processing """
    train = 0
    test = 1
    model_save = 1
    new_size = (30, 30)
    epoch = 150
    color = False    # false for grey img

    # 3 channel batch normalization filter sz cnn layer fcnn
    cnn_filter = [16]
    fc_filter = [16, 1]
    cnn_file = 'h5_files/' + 'cnn_1C_BN_[16]_[16_1]_mean_norm'
    num = 1
    batch_num = 1

    train_path = 'TrainSet/'
    robot_dir = 'robo/'
    no_r_dir = 'no_robo/'

    robo_path = train_path + robot_dir
    no_r_path = train_path + no_r_dir

    # channel
    if color:
        channel = 3
    else:
        channel = 1

    # load data and compress data to new size
    imgs_r, label_r = load_data(robo_path, robot=1)
    imgs_nr, label_nr = load_data(no_r_path, robot=0)

    # mix images and split into batches
    d_train, d_val, d_test = get_data_batch(imgs_r, imgs_nr, label_r, label_nr, batch_num=batch_num, show_data=True)

    """ build model """
    if train:
        # expected 4 d input (none, h, w, channel)
        cnn1 = detect.CNN_model(input_shape=[new_size[0], new_size[1], channel],
                                cnn_filters=cnn_filter,   # if error then img size to small or to many layers
                                drop_out=0.01,
                                fc_filters=fc_filter,
                                opt='adam',                   # or adam
                                loss='binary_crossentropy',      # loss for classification
                                metrics=['acc'])                      # metrics
                                # 'rmsprop'

        cnn1.get_optimizer(lr=1e-4, decay=1e-3)
        cnn1.build_model()
        cnn1.model_compile()
        cnn1.model.summary()

        """ train model """
        # for every batch
        for i in range(batch_num):

            print('___________________________________')
            print('training batch {} starts'.format(i+1))
            # get compressed data
            train_x = get_compressed_imgs(d_train[0][i], data_type='training', size=new_size, color=color)
            print(np.shape(train_x))
            train_y = d_train[1][i]
            # get val data
            val_x = get_compressed_imgs(d_val[0], data_type='validation', size=new_size, color=color)
            val_y = d_val[1]

            # plot image but not more than 10
            plot_img(train_x, d_train[0][i], start=100, end=102)
            # plot_img(train_y, start=10, end=12)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=0.05,
                                                              patience=20,
                                                              verbose=0,
                                                              mode='auto')

            ep = epoch
            bs = 64
            history = cnn1.model.fit(train_x, train_y,
                                     epochs=ep,
                                     validation_data=(val_x, val_y),
                                     batch_size=bs,
                                     #callbacks=[early_stopping],
                                     #callbacks=[lr_decay],
                                     #steps_per_epoch=bs_ep,
                                     #shuffle=True,
                                     verbose=1
                                     )

            cnn1.draw_result(history)


        # save model
        if model_save:
            cnn1.save_model(cnn_file, num)
    # test
    if test:
        bs = 64
        print('testing ...')
        test_x = get_compressed_imgs(d_test[0], data_type='test', size=new_size, color=color)
        test_y = d_test[1]

        # plot image
        plot_img(test_x, d_test[1], start=4, end=6)

        cnn = tf.keras.models.load_model(cnn_file + str(num) + '.h5')
        cnn.summary()
        prediction = cnn.predict(test_x,
                                 batch_size=bs,
                                 verbose=1)

        prediction = np.reshape(prediction, np.shape(d_test[1]))

        for idx, p in enumerate(prediction):
            if p > 0.5:
                prediction[idx] = 1
            else:
                prediction[idx] = 0

        wrong_img_idx = []
        wrong_val = []
        correct_val = []
        for idx, i in enumerate(d_test[1]):
            if prediction[idx] != i:
                wrong_img_idx.append(idx)
                wrong_val.append(prediction[idx])
                correct_val.append(i)

        print('img index with wrong prediction')
        print(wrong_img_idx)
        print('wrong prediction')
        print(wrong_val)
        print('ground truth')
        print(correct_val)
        print('____________________________')
        print('accuracy = {}'.format(1 - len(wrong_val)/len(d_test[1])))

        plt.title('visualization of ground truth label (blue) on top of predicted label (red)')
        plt.subplot(3, 1, 1)
        plt.plot(prediction[:200], 'r')
        plt.plot(d_test[1][:200], 'b')

        plt.subplot(3, 1, 2)
        plt.plot(prediction[200:400], 'r')
        plt.plot(d_test[1][200:400], 'b')

        plt.subplot(3, 1, 3)
        plt.plot(prediction[400:], 'r')
        plt.plot(d_test[1][400:], 'b')

        plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))

