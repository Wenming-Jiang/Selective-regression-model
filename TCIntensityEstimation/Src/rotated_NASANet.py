from __future__ import print_function
import keras
from keras import backend as K
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2,l1,l1_l2
from keras.initializers import TruncatedNormal, RandomNormal, Ones, Constant, Zeros
from keras.utils import np_utils

import os
import pandas as pd
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append("../tools/")
from Evaluation import rotated_evaluation, plot_history, my_evaluation, my_percent_evaluation
from rotate_batch import Data_aug, rotate_by_channel, normalize_data
from read_201_data import read_sita_data, NASANet, read_test_data, Conv32_3fc, Hybrid_Model, New_Hybrid_Model, AlexNet
from read_201_data import Conv32_test
from DAV_pro import DAV, DAV_sample

NASANET = 0

def intensity2class(y):
    y_class = np.zeros(y.shape)
    for i in range(y.shape[0]):
        if y[i] <= 20:   #20
            y_class[i] = 0
        if y[i] > 20 and y[i] <= 33: #20 33
            y_class[i] = 1
        if y[i] > 33 and y[i] <= 63:
            y_class[i] = 2
        if y[i] > 63 and y[i] <= 82:
            y_class[i] = 3
        if y[i] > 82 and y[i] <= 95:
            y_class[i] = 4
        if y[i] > 95 and y[i] <= 112:
            y_class[i] = 5
        if y[i] > 112 and y[i] <= 136:
            y_class[i] = 6
        if y[i] > 136:    #136
            y_class[i] = 7
    return y_class

def train_AlexNet(SITA, CHANNEL, EPOCHS, model_num):
    W_l1RE = 1e-5 # 5e-4 is best
    W_l2RE = 1e-5
    batch_size = 32
    epochs = EPOCHS
    data_augmentation = True
    save_dir = os.path.join(os.getcwd(), 'NASA_model')
    #model_name = 'AlexNetSmallRo' + str(SITA) + '.h5'

    #x_train, y_train, x_test, y_test = read_sita_data(SITA, CHANNEL)
    if model_num == NASANET: 
        #W_l1RE = 5e-4
        W_l1RE = 0
        W_l2RE = 1e-4
        #W_l2RE = 1e-5
        
        #x_train = np.load("../sita_rotated_data/ATLN_2003_2014_data_x_101.npy").astype('float32')
        x_train = np.load("../sita_rotated_data/ATLN_2003_2014_data_x_101_filted.npy").astype('float32')
        y_train = np.load("../sita_rotated_data/ATLN_2003_2014_data_y_201.npy").astype('float32')
        #print(y_train[:10])
        y_train = intensity2class(y_train) #[1,2,3,4,5,6,7,8] 
        print(y_train)
        for i in range(8):
            print(sum(y_train==i))
        y_train = np_utils.to_categorical(y_train)
        print("y_train one hot shape: ", y_train.shape)
        #print(y_train[:10,:])
        
        x_test  = np.load("../sita_rotated_data/ATLN_2015_2016_data_x_101_filted.npy").astype('float32')
        y_test  = np.load("../sita_rotated_data/ATLN_2015_2016_data_y_201.npy").astype('float32')
        x_test = x_test[y_test<=180,:,:,:]
        y_test = y_test[y_test<=180]

        y_test = intensity2class(y_test) #[1,2,3,4,5,6,7,8] 
        print(y_test)
        for i in range(8):
            print(sum(y_test==i))
        y_test = np_utils.to_categorical(y_test)
        print("y_test one hot shape: ", y_test.shape)

        ##V2-450 model get small data(<=55)
        #x_test = np.load("./tmp_0_80_model/ATLN_2015_2016_smalldata_x_101_filted.npy").astype('float32')
        #y_test = np.load("./tmp_0_80_model/ATLN_2015_2016_smalldata_y_101_filted.npy").astype('float32')

        x_train = rotate_by_channel(x_train, np.ones(x_train.shape[0])*SITA, 2)
        x_train = x_train[:, 18:83, 18:83, :]   # 18:83 = 65
        x_train = normalize_data(x_train, x_train.shape[3])

        x_test = x_test[:, 18:83, 18:83, :]   # 18:82 = 64
        x_test = normalize_data(x_test, x_test.shape[3])

        
        model_name_pre = 'AlexNet0-180-'
        print(x_train.shape, x_test.shape)
        model = NASANet(W_l1RE, W_l2RE, (65,65,x_train.shape[3]))
    
    if not data_augmentation:
        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
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
            # fraction of images reserved for validation (strictly between 0 and 1)
            #validation_split=0.0)

        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0)
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=10, min_lr=1e-6)
        #tb = TensorBoard(log_dir='./tmp/log', histogram_freq=10)
        #filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
        filepath="./NASA_model/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint= ModelCheckpoint(filepath, monitor='val_loss', verbose=1,  period = 50)

        history = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True,
                            callbacks=[checkpoint]) #reduce_lr, 

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test RMSE:', np.sqrt(scores[0]))
    print('Test MAE:', scores[1])

    model_name = model_name_pre + str(SITA) + '-' + str(int(scores[1]*10)/10.0) + '.h5'
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
    return y_predict, y_predict_var, dy 



def single_Var_analysis():
    ordered_var_dy = np.load("./Var2015-2017/15/15_10.45.npy").astype('float32')   #change
    #index = [i for i in range(ordered_var_dy.shape[0])]
    #import random
    #random.shuffle(index)
    #np.save("./Var2015-2017/15/" + "var-dy-random_9_24/" + "shuffleindex.npy", index)
    index = np.load("./Var2015-2017/10/" + "var-dy-random_9_30/" + "shuffleindex.npy")   #unchange
    
    middle = int(ordered_var_dy.shape[0] / 2)
    test_index = index[:middle]
    val_index = index[middle:]
    test_index.sort()
    val_index.sort()
    print(len(test_index), len(val_index))

    test_var_dy = []
    val_var_dy = []
    for i in test_index:
        test_var_dy.append(ordered_var_dy[i,:])
    for i in val_index:
        val_var_dy.append(ordered_var_dy[i, :])
    test_var_dy = np.array(test_var_dy)
    val_var_dy = np.array(val_var_dy)
    print(test_var_dy.shape, val_var_dy.shape)

    test_mse = np.zeros((test_var_dy.shape[0],1))
    test_t   = np.zeros((test_var_dy.shape[0],1))
    test_b   = np.zeros((test_var_dy.shape[0],1))
    for i in range(test_var_dy.shape[0]):
        test_mse[i][0] = np.mean(test_var_dy[:,1][:(i+1)]*test_var_dy[:,1][:(i+1)])
        test_t[i][0]   = 489.55 / np.sqrt(i+1) * 1.1   #delta = 0.05, b-a = 20*1.1 * 20*1.1 - 0
        test_b[i][0]   = test_mse[i][0] + test_t[i][0]
    test = np.concatenate((test_var_dy, test_mse, test_t, test_b), axis = 1)

    val_mse = np.zeros((val_var_dy.shape[0],1))
    for i in range(val_var_dy.shape[0]):
        val_mse[i][0] = np.mean(val_var_dy[:,1][:(i+1)]*val_var_dy[:,1][:(i+1)])
    val = np.concatenate((val_var_dy, val_mse), axis = 1)
    
    np.savetxt("./Var2015-2017/15/" + "var-dy-random_9_30/" + "15_val.csv", test, delimiter=',')   #change
    np.savetxt("./Var2015-2017/15/" + "var-dy-random_9_30/" + "15_test.csv", val , delimiter=',')   #change
    print("val_MSE=", test[:,2][-1], "test_MSE=", val[:,2][-1])
    #np.savetxt("ordered_dy.csv", ordered_dy, delimiter=',')
    #np.savetxt("ordered_var.csv", ordered_var, delimiter=',')
    #np.savetxt(str(360/Rotated_Max_Sita)+"-mse.csv", mse, delimiter=',')
    #np.savetxt("t.csv", t, delimiter=',')
    #b = mse + t
    #np.savetxt("b.csv", b, delimiter=',')
    plt.plot(test_var_dy[:,1])
    plt.show()


def multi(total):
    ordered_var_dy = np.load("./Var2015-2017/10/10_10.49.npy").astype('float32')
    index = [i for i in range(ordered_var_dy.shape[0])]
    import random
    random.shuffle(index)

    middle = int(ordered_var_dy.shape[0] / 2)
    test_index = index[:middle]
    val_index = index[middle:]
    test_index.sort()
    val_index.sort()
    #print(len(test_index), len(val_index))

    test_var_dy = []
    val_var_dy = []
    for i in test_index:
        test_var_dy.append(ordered_var_dy[i,:])
    for i in val_index:
        val_var_dy.append(ordered_var_dy[i, :])
    test_var_dy = np.array(test_var_dy)
    val_var_dy = np.array(val_var_dy)
    #print(test_var_dy.shape, val_var_dy.shape)

    test_mse = np.zeros((test_var_dy.shape[0],1))
    test_t   = np.zeros((test_var_dy.shape[0],1))
    test_b   = np.zeros((test_var_dy.shape[0],1))
    for i in range(test_var_dy.shape[0]):
        if i!= test_var_dy.shape[0]-1:
            continue
        test_mse[i][0] = np.mean(test_var_dy[:,1][:(i+1)]*test_var_dy[:,1][:(i+1)])
        test_t[i][0]   = 489.55 / np.sqrt(i+1) * 1.1#(delta=0.01)   #delta = 0.05, b-a = 20*20 - 0
        test_b[i][0]   = test_mse[i][0] + test_t[i][0]
    test = np.concatenate((test_var_dy, test_mse, test_t, test_b), axis = 1)

    val_mse = np.zeros((val_var_dy.shape[0],1))
    for i in range(val_var_dy.shape[0]):
        if i!= val_var_dy.shape[0]-1:
            continue
        val_mse[i][0] = np.mean(val_var_dy[:,1][:(i+1)]*val_var_dy[:,1][:(i+1)])
    val = np.concatenate((val_var_dy, val_mse), axis = 1)
    
    
    if(sum(test[:,4] < val[:,2]) > 0):
        #print(total, "val_b=", test[:,4][-1], "test_MSE=", val[:,2][-1], "True")
        return 1
    #print(total, "val_b=", test[:,4][-1], "test_MSE=", val[:,2][-1])
    return 0    



def ideal_analysis():
    ordered_var_dy = np.load("./Var2015-2017/10/10_10.49.npy").astype('float32')
    #index = [i for i in range(ordered_var_dy.shape[0])]
    #import random
    #random.shuffle(index)
    index = np.load("./Var2015-2017/10/" + "var-dy-random_9_24/" + "shuffleindex.npy")

    middle = int(ordered_var_dy.shape[0] / 2)
    test_index = index[:middle]
    val_index = index[middle:]
    test_index.sort()
    val_index.sort()
    print(len(test_index), len(val_index))

    test_var_dy = []
    val_var_dy = []
    for i in test_index:
        test_var_dy.append(ordered_var_dy[i,:])
    for i in val_index:
        val_var_dy.append(ordered_var_dy[i, :])
    test_var_dy = np.array(test_var_dy)
    val_var_dy = np.array(val_var_dy)
    print(test_var_dy.shape, val_var_dy.shape)

    #test_mse = np.zeros((test_var_dy.shape[0],1))
    ideal_mse = np.zeros((test_var_dy.shape[0],1))
    ideal_ordered_dy = np.sort(abs(test_var_dy[:,1]))
    for i in range(ideal_ordered_dy.shape[0]):
        ideal_mse[i][0] = np.mean(ideal_ordered_dy[:(i+1)]*ideal_ordered_dy[:(i+1)])
    ideal_t   = np.zeros((test_var_dy.shape[0],1))
    ideal_b   = np.zeros((test_var_dy.shape[0],1))
    for i in range(test_var_dy.shape[0]):
        ideal_t[i][0]   = 489.55 / np.sqrt(i+1) * 1.1#(delta=0.01)   #delta = 0.05, b-a = 20*20 - 0
        ideal_b[i][0]   = ideal_mse[i][0] + ideal_t[i][0]
    ideal_ordered_dy = ideal_ordered_dy.reshape(ideal_ordered_dy.shape[0], 1)

    test = np.concatenate((ideal_ordered_dy, ideal_mse, ideal_t, ideal_b), axis = 1)

    ideal_val_mse = np.zeros((val_var_dy.shape[0],1))
    ideal_ordered_val_dy = np.sort(abs(val_var_dy[:,1]))
    #val_mse = np.zeros((val_var_dy.shape[0],1))
    for i in range(val_var_dy.shape[0]):
        ideal_val_mse[i][0] = np.mean(ideal_ordered_val_dy[:(i+1)]*ideal_ordered_val_dy[:(i+1)])
    ideal_ordered_val_dy = ideal_ordered_val_dy.reshape(ideal_ordered_val_dy.shape[0], 1)
    val = np.concatenate((ideal_ordered_val_dy, ideal_val_mse), axis = 1)
    
    np.savetxt("./Var2015-2017/10/" + "var-dy-random_9_29/" + "ideal_val.csv", test, delimiter=',')   #change
    np.savetxt("./Var2015-2017/10/" + "var-dy-random_9_29/" + "ideal_test.csv", val , delimiter=',')   #change

    #plt.plot(val_mse[:,0])
    plt.plot(ideal_mse[10:,0]+ideal_t[10:,0])
    plt.plot(ideal_mse[10:,0])
    plt.plot(ideal_t[10:,0])
    plt.show()


def scalar2img(data):
    data = (data -np.min(data)) / (np.max(data)-np.min(data)) * 255
    return data

def DrawTCMap(pre_str, test_data, y_test, BATCH_SIZE, Rotated_Max_Sita):
    regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2))
    regressmodel.load_weights('../result_model/weightsV2-improvement-450.hdf5')

    y_predict = np.zeros((y_test.shape[0], int(360/Rotated_Max_Sita)))
    for rotatedsita in  range(0, 360, Rotated_Max_Sita):
        testx = rotate_by_channel(test_data, np.ones(test_data.shape[0])*rotatedsita, 2)
        testx = testx[:, 18:83, 18:83, :]

        testx = normalize_data(testx, testx.shape[3])
        y_predict_regress = regressmodel.predict(testx, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
        print("Test data rotated sita: ", rotatedsita, "y_pred: ", y_predict_regress[0])

        img = Image.fromarray(np.uint8(scalar2img(testx[0, :, :, 0]))) 
        img.save(pre_str+str(rotatedsita)+ "_pred" + str(int(y_predict_regress[0] * 10)) +"_IR.png", 'png')

        img = Image.fromarray(np.uint8(scalar2img(testx[0, :, :, 1]))) 
        img.save(pre_str+str(rotatedsita)+ "_pred" + str(int(y_predict_regress[0] * 10)) +"_PMW.png", 'png')

        y_predict[:, int(rotatedsita/Rotated_Max_Sita)] = y_predict_regress

    print(y_predict)
    y_predict_mean = np.mean(y_predict, axis = -1)
    y_predict_var = np.var(y_predict, axis =-1)
    
    img = Image.fromarray(np.uint8(scalar2img(test_data[0, :, :, 0]))) 
    img.save(pre_str+ "_GT" + str(y_test[0]) + "_Mean" + str(int(y_predict_mean*10)) + "_Var" + str(int(y_predict_var*10)) +"_IR.png", 'png')

    img = Image.fromarray(np.uint8(scalar2img(test_data[0, :, :, 1])))
    img.save(pre_str+ "_GT" + str(y_test[0]) + "_Mean" + str(int(y_predict_mean*10)) + "_Var" + str(int(y_predict_var*10)) +"_PMW.png", 'png')

    print("mean: ",         y_predict_mean)
    print("variance: ",     y_predict_var)
    print("ground_truth: ", y_test) 


if __name__ == '__main__':
    #x_test  = np.load("../sita_rotated_data/ATLN_2015_2016_data_x_101_filted.npy").astype('float32')
    #y_test  = np.load("../sita_rotated_data/ATLN_2015_2016_data_y_201.npy").astype('float32')
    #x_test = x_test[y_test<=180,:,:,:]
    #y_test = y_test[y_test<=180]

    #for i in range(10):
    #    index = np.random.randint(0, len(x_test))
    #    test_data = x_test[[index],:,:,:]
    #    print(test_data.shape)
    #    test_y = y_test[[index]]
    #    pre_str ="./visible/ID"+ str(index) + "_"
    #    DrawTCMap(pre_str, test_data, test_y, BATCH_SIZE=32, Rotated_Max_Sita=36)


    x_test  = np.load("../sita_rotated_data/ATLN_2015_2016_data_x_201.npy").astype('float32')
    y_test  = np.load("../sita_rotated_data/ATLN_2015_2016_data_y_201.npy").astype('float32')
    x_test = x_test[y_test<=180,:,:,:]
    y_test = y_test[y_test<=180]
    print(x_test.shape)

    index = 5732
    print("ID: ", index, "GT:", y_test[index])
    pre_str ="./visible/ID"+ str(index) + "_"
    img = Image.fromarray(np.uint8(scalar2img(x_test[index, :, :, 0]))) 
    img.save(pre_str+ "_GT" + str(y_test[index]) + "_IR.png", 'png')
    img = Image.fromarray(np.uint8(scalar2img(x_test[index, :, :, 1])))
    img.save(pre_str+ "_GT" + str(y_test[index]) + "_PMW.png", 'png')

    
    exit()



    #ideal_analysis()   #is OK
    single_Var_analysis()
    exit()

    #single_Var_analysis()
    total = 10000
    out = 0
    for i in range(total):
        if multi(total) == 1:
            out = out + 1

    print(out, total, 100 - (out+0.0)/total * 100)
    exit()

    #x_train = np.load("../sita_rotated_data/ATLN_2003_2014_data_x_101_filted.npy").astype('float32')  # train data set
    #y_train = np.load("../sita_rotated_data/ATLN_2003_2014_data_y_201.npy").astype('float32')

    x_test = np.load("../sita_rotated_data/ATLN_2015_2016_data_x_101_filted.npy").astype('float32')    # evalation data set
    y_test = np.load("../sita_rotated_data/ATLN_2015_2016_data_y_201.npy").astype('float32')
    x_test    = x_test[y_test<=180,:,:,:]
    y_test    = y_test[y_test<=180]
    print(x_test.shape)
    val_x_test = np.load("../sita_rotated_data/ATLN_2017_data_x_101_filted.npy").astype('float32')    # test data set
    val_y_test = np.load("../sita_rotated_data/ATLN_2017_data_y_201.npy").astype('float32')
    x_test = np.concatenate((x_test, val_x_test), axis = 0)
    y_test = np.concatenate((y_test, val_y_test), axis = 0)
    print(x_test.shape)

    #evaluation_rotated
    Rotated_Max_Sita =  24 # 180, 90, 45, 36, 30, 24
    y_predict, y_predict_var, dy = evaluation_rotated(x_test, y_test, 64, Rotated_Max_Sita)
    ordered_indices = np.argsort(y_predict_var)
    ordered_var = y_predict_var[ordered_indices]
    ordered_dy = dy[ordered_indices]
    
    mymse = np.mean(ordered_dy*ordered_dy)
    ordered_var = ordered_var.reshape(ordered_var.shape[0], 1)
    ordered_dy  = ordered_dy.reshape(ordered_dy.shape[0], 1)
    ordered_var_dy = np.concatenate((ordered_var, ordered_dy), axis=1)
    np.save("./Var2015-2017/"+str(int(360/Rotated_Max_Sita))+"/" + str(int(360/Rotated_Max_Sita)) +"_" + str(int(np.sqrt(mymse)*100)/100.0) + ".npy" , ordered_var_dy)







    '''
    y_class_predict, y_class, dy = evaluation(x_test, y_test, 64)
    pred_class = y_class_predict.argmax(axis=-1)
    probability = np.max(y_class_predict, axis=-1)
    ordered_indices = np.argsort(probability)

    ordered_probability = probability[ordered_indices]  #7567
    ordered_pred_class = pred_class[ordered_indices]    #7567
    ordered_y_class  = y_class[ordered_indices]         #7567
    ordered_dy = dy[ordered_indices]                    #7567

    begin = 3000
    print(sum(ordered_pred_class[begin:]==ordered_y_class[begin:])/(7567-begin))
    np.sqrt(np.mean(dy[begin:]*dy[begin:]))
    trueclassdy = ordered_dy[ordered_pred_class==ordered_y_class]
    exit()
    #RMSE:(132, 11.489) (131, 11.445) (133, 11.53) (135, 11.618) (136, 11.662) (137, 11.70)
    train_AlexNet(0, [0,3], 200, NASANET)    # RMSE: 11.4

    #train_AlexNet(288, [0,3], 200, WEAKALEXNET)
    #train_AlexNet(324, [0,3], 200, WEAKALEXNET)

    '''

        #mse = np.zeros(dy.shape)
    #t   = np.zeros(dy.shape)
    #for i in range(dy.shape[0]):
    #    mse[i] = np.mean(ordered_dy[:(i+1)]*ordered_dy[:(i+1)])
        #t[i]   = np.sqrt(97291.55/(i+1))
    #np.savetxt("ordered_dy.csv", ordered_dy, delimiter=',')
    #np.savetxt("ordered_var.csv", ordered_var, delimiter=',')
    #np.savetxt(str(360/Rotated_Max_Sita)+"-mse.csv", mse, delimiter=',')
    #np.savetxt("t.csv", t, delimiter=',')
    #b = mse + t
    #np.savetxt("b.csv", b, delimiter=',')
    #plt.plot(ordered_dy)
    #plt.show()

















































'''
    if model_num == WEAKALEXNET: 
        W_l1RE = 0
        W_l2RE = 1e-5
        save_dir = os.path.join(os.getcwd(), 'weakAlexnet_models')
        #x_train = np.load("../sita_rotated_data/ATLN_2003_2014_data_x_65_filted_norm.npy").astype('float32')
        #x_train = np.load("../sita_rotated_data/ATLN_2003_2014_data_x_65_filted.npy").astype('float32')
        x_train = np.load("../sita_rotated_data/ATLN_2003_2014_data_x_101_filted.npy").astype('float32')
        x_test  = np.load("../sita_rotated_data/ATLN_2015_2016_data_x_65_filted_norm.npy").astype('float32')
        y_train = np.load("../sita_rotated_data/ATLN_2003_2014_data_y_201.npy").astype('float32')
        y_test  = np.load("../sita_rotated_data/ATLN_2015_2016_data_y_201.npy").astype('float32')
        
        x_train = rotate_by_channel(x_train, np.ones(x_train.shape[0])*SITA, 2)
        x_train = x_train[:, 18:83, 18:83, :]

        new_x_train = []
        new_y_train = []
        for i in range(x_train.shape[0]):
            #if y_train[i] <= 140 and y_train[i] >= 35:
            if y_train[i] <= 70 :
                new_x_train.append(x_train[i,:,:,:])
                new_y_train.append(y_train[i])
        x_train = np.array(new_x_train)
        y_train = np.array(new_y_train)

        new_x_train = []
        new_y_train = []
        for i in range(x_test.shape[0]):
            if y_test[i] <= 70:
            #if y_test[i] <= 140 and y_test[i] >= 35:
                new_x_train.append(x_test[i,:,:,:])
                new_y_train.append(y_test[i])
        x_test = np.array(new_x_train)
        y_test = np.array(new_y_train)

        x_train = normalize_data(x_train, len(CHANNEL)) 
        model_name = 'weakAlexNet' + str(SITA) + '.h5'
        print(x_train.shape, x_test.shape)
        model = weakAlexNet(W_l1RE, W_l2RE, (65,65,len(CHANNEL)))
'''