from PIL import Image
import numpy as np
import pandas as pd
from PIL import Image
import h5py

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

def load_data(data_path):
    #data_path = "../data/TCIR-ALL_2017.h5"
    print("begin read data from "+ data_path +" ...")
    #load "info" as pandas dataframe
    data_info = pd.read_hdf(data_path, key="info", mode='r')
    #load "matrix" as numpy ndarray, this could take longer time
    with h5py.File(data_path, 'r') as hf:
        data_matrix = hf['matrix'][:]
        #print(data_matrix.shape)    
    return data_matrix, data_info

def pre_processing(data_path):
    #data_path1 = "../data/TCIR-ALL_2017.h5"
    data_x, data_info = load_data(data_path)
    data_info = data_info.values
    data_y = data_info[:,5]  # Vmax
    #return X, Y, The data type of both are np.ndarray.
    # X:(None, 201, 201, 4) = (None, 64, 64, 4) [IR WV VIS PMW]
    # Y:(None, 1)
    #data_x = data_x[:, 68:133, 68:133, :]  # for the 65 * 65
    data_x = np.nan_to_num(data_x)
    data_x[data_x>1000] = 0
    return data_x, data_y

def pre_processing2(first_time, second_time, data_path):
    data_x, data_info = load_data(data_path)    
    data_info = data_info.values 
    # data_set ID lon lat time Vmax R35_4qAVG MSLP
    data_time = data_info[:,4] # time
    data_time = data_time.astype('int')
    data_y = data_info[:,5]  # Vmax
    #return X, Y, The data type of both are np.ndarray.
    # X:(None, 201, 201, 4) = (None, 64, 64, 4)
    # Y:(None, 1)
    new_data_x = []
    new_data_y = []
    #data_x = data_x[:, 68:133, 68:133, :]  # for the 64 * 64
    data_x = np.nan_to_num(data_x)
    data_x[data_x>1000] = 0
    for i in range(len(data_time)):
        if (data_time[i] >= first_time) & (data_time[i]< second_time):
            new_data_x.append(data_x[i,:,:,:])
            new_data_y.append(data_y[i])
    return np.array(new_data_x), np.array(new_data_y)

def normalize_data(x_test, chanel_num):
    result=[]
    height = x_test.shape[1]
    for each_sam in x_test:
        new_sam = [];
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

def rotated_evaltion(test_data, test_Y, model, BATCH_SIZE):
    #model = load_model('./saved_models/keras_5_3best_400trained_model.h5')
    predict_Y = model.predict(test_data, batch_size=BATCH_SIZE, verbose=0)
    predict_Y = predict_Y.reshape(-1)

    length = len(test_Y)
    print("no rotated mae:" + str(np.mean(np.abs(predict_Y[:length] - test_Y))))
    rotated_num = int(len(predict_Y)/length)

    result = np.zeros(length)
    tmp_predict_Y = predict_Y
    for i in range(rotated_num):
        print(np.mean(np.abs(tmp_predict_Y[:length] - test_Y)))
        result = result + (tmp_predict_Y[:length] - test_Y)
        tmp_predict_Y = tmp_predict_Y[length:]
    result = result/rotated_num

    mae = np.mean(np.abs(result)) # MAE
    print(str(rotated_num) + " rotated mae: " + str(mae))
    return mae


def fenkaishuju():
    #train_data_path_CPAC = "../data/TCIR-CPAC_IO_SH.h5"       # CPAC,IO,SH     14.6GB data
    train_data_path_ATLN = "../data/TCIR-ATLN_EPAC_WPAC.h5"   # ATLN,EPAC,WPAC 30GB   data
    #x_train, y_train = pre_processing(train_data_path_CPAC)
    x_train, y_train = pre_processing2(2000000000, 2015000000, train_data_path_ATLN)
    print(x_train.shape)
    print(y_train.shape)
    np.save("../rotated_data/ATLN_2003_2014_data_x_201.npy", x_train)
    np.save("../rotated_data/ATLN_2003_2014_data_y_201.npy", y_train)


    x_test, y_test = pre_processing2(2015000000, 2017000000, train_data_path_ATLN)
    print(x_test.shape)
    print(y_test.shape)
    np.save("../rotated_data/ATLN_2015_2016_data_x_201.npy", x_test)
    np.save("../rotated_data/ATLN_2015_2016_data_y_201.npy", y_test)

    print("OK!!!")
#x_test = np.load("../data/ATLN_2015_2016_data_x.npy")
#new_x_test = normalize_data(x_test, 4)
#np.save("../norm_data_65/ATLN_2015_2016_data_x.npy", new_x_test)

#x_train = np.load("../data/ATLN_2003_2014_data_x.npy")
#new_x_train = normalize_data(x_train, 4)
#np.save("../norm_data_65/ATLN_2003_2014_data_x.npy", new_x_train)

if __name__ == '__main__':

    #x_train = np.load("../rotated_data/ATLN_2015_2016_data_x_201.npy").astype("float32")
    #y_train = np.load("../rotated_data/ATLN_2015_2016_data_y_201.npy").astype("float32")
    #save_rotated_data()



