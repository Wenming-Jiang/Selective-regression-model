import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def evaluation_rotated(test_data, y_test, BATCH_SIZE, Rotated_Max_Sita):
    #classmodel = load_model('./NASA_model/AlexNet0-180-0-0.5.h5')
    regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2))
    regressmodel.load_weights('D:/JWM_ZHOUZH/JiangWenMing/NASAwork/result_model/weightsV2-improvement-450.hdf5')

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


x_test  = np.load("../sita_rotated_data/ATLN_2015_2016_data_x_101_filted.npy").astype('float32')
y_test  = np.load("../sita_rotated_data/ATLN_2015_2016_data_y_201.npy").astype('float32')
x_test = x_test[y_test<=180,:,:,:]
y_test = y_test[y_test<=180]
