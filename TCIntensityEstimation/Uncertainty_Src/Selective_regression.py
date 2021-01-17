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
from PIL import Image

import sys
sys.path.append("../tools/")

#regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2))
#regressmodel.load_weights('../result_model/weightsV2-improvement-450.hdf5')
#regressmodel.load_weights('./NASA_model/weights-improvement-250-131.65.hdf5')

def uncertainty_loss(y_true, y_pred, alpha=0.5, beta=0.1, k=5.0): 
    # y_pred.shape y_true.shape
    print("************************** [JWM] ******************************")
    print(K.shape(y_true), K.shape(y_pred))
    #loss_regular = K.mean(K.square(y_pred[:,0] - y_true[:,0])) ##MSE
    loss_regular = K.mean(K.abs(y_pred[:,0] - y_true[:,0])) ##MAE
    #return loss_regular
    #loss_constrain1 = K.mean(k * (K.exp((y_true - y_pred[:,1])/4) + K.exp((y_pred[:,2] - y_true)/4)), axis=-1)
    loss_constrain1 = K.mean(K.maximum(y_true[:,0] - y_pred[:,1] + 2, 0)**2 + K.maximum(y_pred[:,2] - y_true[:,0] - 2 , 0)**2)
    loss_constrain2 = K.mean(K.square(y_pred[:,1] - y_pred[:,2]))
    loss_total = alpha * loss_regular + k * loss_constrain1 + beta * loss_constrain2
    return loss_total

def HoverY(y_true, y_pred):
    return K.mean((y_pred[:,1] > y_true[:,0]))

def realRMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred[:,0] - y_true[:,0])))

def LlessY(y_true, y_pred):
    return K.mean((y_true[:,0] > y_pred[:,2]))

def H_L_range(y_true, y_pred):
    return K.mean(K.abs(y_pred[:,1] - y_pred[:,2]))

def plot_history(history, fig_name, ignore_num=0, show = False):
    import matplotlib.pyplot as plt
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    acc_values = history_dict['mean_absolute_error']
    val_acc_values = history_dict['val_mean_absolute_error']

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

def AlexNet(W_l1RE, W_l2RE, shape):

    input = Input(shape=shape)
    curr = Conv2D(16, (4, 4), strides=2, padding='valid', 
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(input)
    curr = Activation('relu')(curr)
    
    curr = Conv2D(32, (3, 3), strides = 2,
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(curr)
    curr = Activation('relu')(curr)
    curr = BatchNormalization(axis=3)(curr)

    curr = Conv2D(64, (3, 3), strides = 2,
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(curr)
    curr = Activation('relu')(curr)
    
    curr = Conv2D(128, (3, 3), strides = 2,
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(curr)
    curr = Activation('relu')(curr)
    curr = BatchNormalization(axis=3)(curr)

    curr = Flatten()(curr)
    curr = Dense(256, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(curr)
    curr = Activation('relu')(curr)

    # auxiliary output: y^{hat}
    curr1 = Dense(64, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(curr)
    curr1 = Activation('relu')(curr1)
    curr1 = Dense(1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(curr1)
    curr1 = Activation('linear')(curr1)
    
    ## sup limit: H
    curr2 = Dense(64, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(curr)
    curr2 = Activation('relu')(curr2)
    curr2 = Dense(1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(curr2)
    curr2 = Activation('linear')(curr2)

    ## inf limit: L
    curr3 = Dense(64, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(curr)
    curr3 = Activation('relu')(curr3)
    curr3 = Dense(1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(curr3)
    curr3 = Activation('linear')(curr3)

    final_output = Concatenate(axis=1, name="selective_output")([curr1, curr2, curr3])

    model = Model(inputs=input, outputs=final_output)

    #opt = keras.optimizers.rmsprop(lr=0.001)
    opt = keras.optimizers.rmsprop(lr=0.005)

    # Let's train the model using RMSprop
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae', HoverY, LlessY, H_L_range])
    model.compile(loss=uncertainty_loss, optimizer=opt, metrics=[realRMSE, HoverY, LlessY, H_L_range])
    model.summary()
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
        filepath="./result_model/uncertainty_loss-{epoch:02d}-{val_loss:.2f}.hdf5"
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
                            callbacks=[reduce_lr, checkpoint])

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


def HoverY1(y_true, y_pred):
    return np.mean((y_pred[:,1] > y_true[:,0]))

def realRMSE1(y_true, y_pred):
    return np.sqrt(np.mean((y_pred[:,0] - y_true[:,0])**2))

def LlessY1(y_true, y_pred):
    return np.mean((y_true[:,0] > y_pred[:,2]))

def H_L_range1(y_true, y_pred):
    return np.mean(np.abs(y_pred[:,1] - y_pred[:,2]))

def YinL_H(y_true, y_pred):
    return ((y_true[:,0]>=y_pred[:,2]) * (y_true[:,0]<=y_pred[:,1])).mean()

def YpinYt2std(y_true, y_pred, std):
    return ((y_pred[:,0]>=y_true[:,0]-std) * (y_pred[:,0]<=y_true[:,0]+std)).mean()

def Selective_evaluation_rotated(testset_xpath, testset_ypath, modelpath, BATCH_SIZE, Rotated_Max_Sita):
    regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2))
    #regressmodel.load_weights('../result_model/weightsV2-improvement-450.hdf5')
    regressmodel.load_weights(modelpath)

    x_test  = np.load(testset_xpath).astype('float32')
    y_test  = np.load(testset_ypath).astype('float32')
    x_test = x_test[y_test<=180,:,:,:]
    y_test = y_test[y_test<=180]
    y_test = y_test[:, np.newaxis]

    #Rotated_Max_Sita = 45
    y_predict = np.zeros((int(360/Rotated_Max_Sita), y_test.shape[0], 3))

    for rotatedsita in  range(0, 360, Rotated_Max_Sita):
        testx = rotate_by_channel(x_test, np.ones(x_test.shape[0])*rotatedsita, 2)
        testx = testx[:, 18:83, 18:83, :] # 18:82 = 64
        testx = normalize_data(testx, testx.shape[3])

        y_predict_regress = regressmodel.predict(testx, batch_size=BATCH_SIZE, verbose=0) # [y, H, L]
        thisrmse = realRMSE1(y_test, y_predict_regress)
        #print(y_predict_regress.shape)
        print(f'------------- Test data rotated sita: {rotatedsita} -----------------------')
        print(f'/*** 2*realRMSE : {2*thisrmse*1000//1/1000} ****')
        print(f'/****  MAE(H-L) : {H_L_range1(y_test, y_predict_regress)*1000//1/1000} ****')
        print(f'/****   H > Y   : {HoverY1(y_test, y_predict_regress)*1000//1/10}% ****')
        print(f'/****   L < Y   : {LlessY1(y_test, y_predict_regress)*1000//1/10}% ****')
        print(f'/** Y_t in 2*STD: {YpinYt2std(y_test, y_predict_regress[:,[0]], thisrmse)*1000//1/10}% ****')
        print(f'/** Y_t in [L, H]: {YinL_H(y_test, y_predict_regress)*1000//1/10}% ****')
        print(f'/** Y_p in [L, H]: {YinL_H(y_predict_regress[:,[0]], y_predict_regress)*1000//1/10}% ****')
        print(f'---------------------------------------------------------------------------')
        #y_predict[:, int(rotatedsita/Rotated_Max_Sita)] = y_predict_regress[:,0]
        y_predict[int(rotatedsita/Rotated_Max_Sita), :, :] = y_predict_regress

    y_predict_mean = np.mean(y_predict, axis = 0)
    #print(y_predict_mean.shape)
    y_predict_var = np.var(y_predict, axis = 0)

    rmse = np.sqrt(np.mean((y_predict_mean[:,0]-y_test[:,0])**2))
    print(f'------------- Total rotated number:{int(360/Rotated_Max_Sita)}, Summary as fellow -------------------')
    print(f'Total - rotated blend RMSE: {rmse*1000//1/1000}')
    print(f'Use Selective: y in range[{2*rmse*1000//1/1000}]')
    print(f'Use MaxMin   : y in range[{H_L_range1(y_test, y_predict_mean)*1000//1/1000}]')
    print(f'--  MAE(H-L) : {H_L_range1(y_test, y_predict_mean)*1000//1/1000}  ------')
    print(f'--   H > Y   : {HoverY1(y_test, y_predict_mean)*1000//1/10}%  ------')
    print(f'--   L < Y   : {LlessY1(y_test, y_predict_mean)*1000//1/10}%  ------')
    print(f'- Y_t in 2*STD: {YpinYt2std(y_test, y_predict_mean[:,[0]], rmse)*1000//1/10}%  ------')
    print(f'- Y_t in [L, H]: {YinL_H(y_test, y_predict_mean)*1000//1/10}%  ------')
    print(f'- Y_p in [L, H]: {YinL_H(y_predict_mean[:,[0]], y_predict_mean)*1000//1/10}%  ------')

    sorted_index = np.argsort(y_test[:,0])
    y_test = y_test[sorted_index, :]
    y_predict_mean = y_predict_mean[sorted_index, :]
    num = np.arange(y_test.shape[0])
    plt.figure(figsize=(30, 8), dpi=300)
    plt.plot(num, y_test, 'bo', label='y_true')
    plt.plot(num, y_predict_mean[:,0], 'g^', label='y_pred')
    plt.plot(num, y_predict_mean[:,1], 'r*', label='High')
    plt.plot(num, y_predict_mean[:,2], 'y*', label='Low')
    plt.show()
    import time
    modelname = modelpath.split("/")[-1]
    figurename = f'./Output/{modelname}_{time.strftime("%Y-%m-%d_%H-%M", time.localtime())}.png'
    print(f'The y,H,L vs number figure Saved as: {figurename}')
    plt.savefig(figurename, format='png', bbox_inches = 'tight')
    
    return 0
    dy = y_predict_mean - y_test
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
    #parser.parse_args()

    parser.add_argument("-P", "--datapath", default="../Data/TCIR-ATLN_EPAC_WPAC.h5", help="the TCIR dataset file path")
    parser.add_argument("-Tx", "--trainset_xpath", default="../Data/ATLN_2003_2014_data_x_101.npy", help="the trainning set x file path")
    parser.add_argument("-Ty", "--trainset_ypath", default="../Data/ATLN_2003_2014_data_y_101.npy", help="the trainning set y file path")

    parser.add_argument("-Tex", "--testset_xpath", default="../Data/ATLN_2015_2016_data_x_101.npy", help="the test set x file path")
    parser.add_argument("-Tey", "--testset_ypath", default="../Data/ATLN_2015_2016_data_y_101.npy", help="the test set y file path")

    parser.add_argument("-E", "--epoch", default=600, help="epochs for trainning")
    parser.add_argument("--trainEnable", default=False, help="train or not. If 'True'->train, 'False'->test")
    parser.add_argument("--modelPath", default="./result_model/uncertainty_loss-300-242.16.hdf5", help="train ready model path")
    args = parser.parse_args()
    
    if not args.trainEnable:
        if args.modelPath == "":
            print("You must provide 'the model path', because 'trainEnable==Flase'")
            exit()
        Selective_evaluation_rotated(args.testset_xpath, args.testset_ypath, args.modelPath, BATCH_SIZE=64, Rotated_Max_Sita=45)
    else:
        train_AlexNet(300, args.trainset_xpath, args.trainset_ypath, args.testset_xpath, args.testset_ypath)
