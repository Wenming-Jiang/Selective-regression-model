from __future__ import print_function
import keras
from keras import backend as K
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras import models
from keras import layers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2,l1,l1_l2
from keras.initializers import TruncatedNormal, RandomNormal
from keras.utils import np_utils
#from keras.layers.core import Lambda

import os
#import pandas as pd
import numpy as np
#import h5py
import math
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append("../tools/")

#regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2))
#regressmodel.load_weights('../result_model/weightsV2-improvement-450.hdf5')
#regressmodel.load_weights('./NASA_model/weights-improvement-250-131.65.hdf5')

def plot_history(history, fig_name, ignore_num=0, show = False):
    import matplotlib.pyplot as plt
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    acc_values = history_dict['mean_squared_error']
    val_acc_values = history_dict['val_mean_squared_error']

    epochs = range(1, len(loss_values) + 1 -ignore_num)

    plt.plot(epochs, loss_values[ignore_num:], 'bo', label='Training loss')#bo:blue dot蓝点
    plt.plot(epochs, val_loss_values[ignore_num:], 'ro', label='Validation loss')#b: blue蓝色
    #plt.plot(epochs, acc_values[ignore_num:], 'b', label='Training mae')#bo:blue dot蓝点
    #plt.plot(epochs, val_acc_values[ignore_num:], 'r-', label='Validation mae')#b: blue蓝色
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig = plt.gcf() # plt.savefig(fig_name)
    if show == True:
        plt.show()
    fig.savefig(fig_name, dpi=100)
    plt.close()

def rotate_by_channel(data, sita, length=2):
    newdata = []
    chanel_num = data.shape[3]
    height = data.shape[1]
    if length > 1:
        for index, singal in enumerate(data):
            new_sam = np.array([])
            for i in range(chanel_num):
                channel = singal[:,:,i]
                img = Image.fromarray(channel)
                new_img = img.rotate(sita[index])
                new_channel = np.asarray(new_img)
                if i==0:
                    new_sam = new_channel
                else:
                    new_sam = np.concatenate((new_sam, new_channel), axis = 1) 
            new_sam = new_sam.reshape((height,height,chanel_num),order='F')
            newdata.append(new_sam)
    else:
        print("Error! data length = 1...")
    return np.array(newdata)

def mean_squared_error2(y_true, y_pred):
    
    return loss

def AlexNet(W_l1RE, W_l2RE, shape):
    model = Sequential() # 16 32 64 128    256 64 1  
    model.add(Conv2D(16, (4, 4), strides = 2, padding='valid',
                     input_shape=shape, 
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    #model.add(AveragePooling2D((2, 2), strides = 1))
    model.add(Conv2D(32, (3, 3), strides = 2, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(64, (3, 3), strides = 2, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3) , strides = 2, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), 
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=3))

    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Dense(1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    #model.add(Activation('softmax'))
    model.add(Activation('linear'))

    #opt = keras.optimizers.rmsprop(lr=0.001)
    opt = keras.optimizers.rmsprop(lr=0.005)

    # Let's train the model using RMSprop
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    #model.summary()
    return model

def normalize_data(x_test, chanel_num):
    result=[]
    height = x_test.shape[1]
    for each_sam in x_test:
        new_sam = []
        for i in range(chanel_num):
            chanel = each_sam[:,:,i]
            chanel = (chanel - np.mean(chanel)) / (np.std(chanel)+0.01)
            if i==0:
                new_sam = chanel
            else:
                new_sam = np.concatenate((new_sam, chanel), axis =1)
               
        new_sam = new_sam.reshape((height,height,chanel_num),order='F')
        result.append(new_sam)
    result = np.array(result)
    return result

def train_AlexNet(EPOCHS, trainset_xpath, trainset_ypath, testset_xpath, testset_ypath):
    W_l1RE = 1e-5 # 5e-4 is best
    W_l2RE = 1e-5
    batch_size = 64
    epochs = EPOCHS
    data_augmentation = True
    save_dir = os.path.join(os.getcwd(), 'result_model')
    #model_name = 'AlexNetSmallRo' + str(SITA) + '.h5'

    #W_l1RE = 5e-4
    W_l1RE = 0
    W_l2RE = 1e-4
    #W_l2RE = 1e-5
    #x_train = np.load("../data/ATLN_2003_2014_data_x_101.npy").astype('float32')
    #y_train = np.load("../data/ATLN_2003_2014_data_y_201.npy").astype('float32')
    #x_test  = np.load("../data/ATLN_2015_2016_data_x_101.npy").astype('float32')
    #y_test  = np.load("../data/ATLN_2015_2016_data_y_201.npy").astype('float32')
    
    x_train = np.load(trainset_xpath).astype('float32')
    y_train = np.load(trainset_ypath).astype('float32')
    x_test  = np.load(testset_xpath).astype('float32')
    y_test  = np.load(testset_ypath).astype('float32')

    x_test = x_test[y_test<=180,:,:,:]
    y_test = y_test[y_test<=180]
    x_train = x_train[:, 18:83, 18:83, :]   # 18:83 = 65
    x_train = normalize_data(x_train, x_train.shape[3])
    x_test = x_test[:, 18:83, 18:83, :]   # 18:82 = 64
    x_test = normalize_data(x_test, x_test.shape[3])

    print("the shape of train set and test set: ", x_train.shape, x_test.shape)
    model_name_pre = 'Sel_PostNet-'
    model = AlexNet(W_l1RE, W_l2RE, x_train.shape[1:])

        #model.load_weights('./NASA_model/weights-improvement-200-148.05.hdf5')
    if not data_augmentation:
        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
    
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=20, min_lr=1e-6)
        #tb = TensorBoard(log_dir='./tmp/log', histogram_freq=10)
        filepath="./NASA_model/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint= ModelCheckpoint(filepath, monitor='val_loss', verbose=1,  period = 50)
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=0,  # epsilon for ZCA whitening
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None)#,
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),#Mygen(x_train, y_train, batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True,
                            steps_per_epoch=int(x_train.shape[0]/batch_size)+1,
                            callbacks=[reduce_lr])

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test RMSE:', np.sqrt(scores[0]))
    print('Test MAE:', scores[1])
    model_name = model_name_pre + '-RMSE' + str(int(scores[0]*100)/100.0) + '.h5'
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    #plot_history(history, model_path+"-"+str(int(np.sqrt(scores[0])*100)/100.0)+".png", 100) # plot_history(history=history, ignore_num=5)
    plot_history(history, model_path+"-"+str(int(scores[1]*10)/10.0)+".png", 100) # plot_history(history=history, ignore_num=5)
    print("Over!!!")

def evaluation(test_data, y_test, BATCH_SIZE):
    classmodel = load_model('./NASA_model/AlexNet0-180-0-0.5.h5')
    regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2))
    regressmodel.load_weights('../result_model/weightsV2-improvement-450.hdf5')

    Rotated_Max_Sita = 45
    y_predict = np.zeros(y_test.shape)
    y_class_predict = np.zeros((y_test.shape[0], 8))

    for rotatedsita in  range(0, 360, Rotated_Max_Sita):
        testx = rotate_by_channel(test_data, np.ones(test_data.shape[0])*rotatedsita, 2)
        testx = testx[:, 18:83, 18:83, :]
        testx = normalize_data(testx, testx.shape[3])

        y_predict_regress = regressmodel.predict(testx, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
        print("Test data rotated sita: ", rotatedsita)
        y_predict = y_predict + y_predict_regress
        rmse = np.sqrt(np.mean((y_predict_regress-y_test) * (y_predict_regress-y_test)))
        print(str(rotatedsita/Rotated_Max_Sita+1) + "- rotated blend RMSE: " + str(rmse))

        y_class_predict_tmp = classmodel.predict(testx, batch_size=32, verbose=0)
        if(len(y_class_predict_tmp.shape)==3):
            y_class_predict_tmp = y_class_predict_tmp.reshape(-1)
        y_class_predict = y_class_predict + y_class_predict_tmp
    
    y_predict = y_predict / (360/Rotated_Max_Sita)
    rmse = np.sqrt(np.mean((y_predict-y_test) * (y_predict-y_test)))
    print("Total - rotated blend RMSE: " + str(rmse))

    y_class_predict = y_class_predict / (360/Rotated_Max_Sita)
    dy = y_predict - y_test
    y_class = intensity2class(y_test)

    #np.savetxt("y_class_predict.csv", y_class_predict, delimiter=',')
    #np.savetxt("y_class_predict_maxindex.csv", y_class_predict.argmax(axis=-1), delimiter=',')
    #np.savetxt("y_class.csv", y_class, delimiter=',')
    #np.savetxt("dy.csv", dy, delimiter=',')
    return y_class_predict, y_class, dy

def evaluation_rotated(test_data, y_test, BATCH_SIZE, Rotated_Max_Sita):
    #classmodel = load_model('./NASA_model/AlexNet0-180-0-0.5.h5')
    regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2))
    regressmodel.load_weights('../result_model/weightsV2-improvement-450.hdf5')

    #Rotated_Max_Sita = 45
    y_predict = np.zeros((y_test.shape[0], int(360/Rotated_Max_Sita)))

    for rotatedsita in  range(0, 360, Rotated_Max_Sita):
        testx = rotate_by_channel(test_data, np.ones(test_data.shape[0])*rotatedsita, 2)
        testx = testx[:, 18:83, 18:83, :]
        testx = normalize_data(testx, testx.shape[3])

        y_predict_regress = regressmodel.predict(testx, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
        print("Test data rotated sita: ", rotatedsita)
        y_predict[:, int(rotatedsita/Rotated_Max_Sita)] = y_predict_regress
        rmse = np.sqrt(np.mean((y_predict_regress-y_test) * (y_predict_regress-y_test)))
        print(str(rotatedsita/Rotated_Max_Sita+1) + "- rotated blend RMSE: " + str(rmse))

    y_predict_mean = np.mean(y_predict, axis = -1)
    y_predict_var = np.var(y_predict, axis =-1)
    rmse = np.sqrt(np.mean((y_predict_mean-y_test) * (y_predict_mean-y_test)))
    print("Total - rotated blend RMSE: " + str(rmse))

    dy = y_predict_mean - y_test

    #np.savetxt("y_class_predict.csv", y_class_predict, delimiter=',')
    #np.savetxt("y_class_predict_maxindex.csv", y_class_predict.argmax(axis=-1), delimiter=',')
    #np.savetxt("y_class.csv", y_class, delimiter=',')
    #np.savetxt("dy.csv", dy, delimiter=',')
    sorted_index = np.argsort(y_predict_var)
    total_num = len(sorted_index)
    #sorted_x_data = test_data[sorted_index,:,:,:]
    #sorted_y_data = y_test[sorted_index]
    sorted_dy = dy[sorted_index]
    epochs = np.arange(total_num)
    #plt.plot(epochs, np.sort(np.abs(dy)), 'bo', label='Training loss')
    #plt.show()
    plt.plot(epochs, sorted_dy, 'bo', label='Training loss')#bo:blue dot蓝点
    plt.show()
    #np.save(sorted_index_filename, sorted_index)
    #np.savetxt("y_class_predict.csv", y_class_predict, delimiter=',')
    #np.savetxt("y_class_predict_maxindex.csv", y_class_predict.argmax(axis=-1), delimiter=',')
    #np.savetxt("y_class.csv", y_class, delimiter=',')
    #np.savetxt("dy.csv", dy, delimiter=',')
    return y_predict, y_predict_var, dy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()

    parser.add_argument("-P", "--datapath", default="../data/TCIR-ATLN_EPAC_WPAC.h5", help="the TCIR dataset file path")
    parser.add_argument("-Tx", "--trainset_xpath", default="../data/ATLN_2003_2014_data_x_101.npy", help="the trainning set x file path")
    parser.add_argument("-Ty", "--trainset_ypath", default="../data/ATLN_2003_2014_data_y_101.npy", help="the trainning set y file path")

    parser.add_argument("-Tex", "--testset_xpath", default="../data/ATLN_2015_2016_data_x_101.npy", help="the test set x file path")
    parser.add_argument("-Tey", "--testset_ypath", default="../data/ATLN_2015_2016_data_y_101.npy", help="the test set y file path")

    parser.add_argument("-E", "--epoch", default=600, help="epochs for trainning")
    args = parser.parse_args()
    
    train_AlexNet(600, args.trainset_xpath, args.trainset_ypath, args.testset_xpath, args.testset_ypath)



