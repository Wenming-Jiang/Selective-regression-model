import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluation_MCDropout():
    #npyname = "./test_predict_age_MAE5.21.npy"
    npyname = "./val_predict_age_MAE5.37.npy"
    test_predict_age = np.load(npyname).astype('float32')
    print(test_predict_age.shape)
    y_predict = np.mean(test_predict_age[:,1:], axis = -1)
    y_true = test_predict_age[:,0]
    print(y_predict.shape, y_true.shape)

    dy = y_predict - y_true
    y_predict_var = np.var(test_predict_age[:,1:], axis=-1)
    return y_predict, y_predict_var, dy

def preduce_txt():
    csvfilename = "val.csv" # or "test.csv"
    txtfilename = "val.txt" # or "test.txt"
    csv_data = pd.read_csv(csvfilename)
    fp = open(txtfilename, 'w')
    data = np.array(csv_data)  #(1078L, 1L)
    for i in range(data.shape[0]):
        row = data[i,0]
        rowsp = row.split(";")
        fp.write(rowsp[0]+" "+rowsp[1]+"\n")
    fp.close()

if __name__ == '__main__':
    y_predict, y_predict_var, dy = evaluation_MCDropout()
    ordered_indices = np.argsort(y_predict_var)
    ordered_var = y_predict_var[ordered_indices]
    ordered_dy = dy[ordered_indices]
    plt.plot(ordered_dy)
    #plt.show()
    mae = np.zeros(dy.shape)
    t   = np.zeros(dy.shape)
    for i in range(dy.shape[0]):
        #mse[i] = np.mean(ordered_dy[:(i+1)]*ordered_dy[:(i+1)])
        mae[i] = np.mean(abs(ordered_dy[:(i+1)]))
        t[i]   = np.sqrt(38004.51/(i+1))

    mydir = "./MCDropout_val/"   # "./MCDropout_test/"
    np.savetxt(mydir+"ordered_dy.csv", ordered_dy, delimiter=',')
    np.savetxt(mydir+"ordered_var.csv", ordered_var, delimiter=',')
    np.savetxt(mydir+"mae.csv", mae, delimiter=',')
    np.savetxt(mydir+"t.csv", t, delimiter=',')
    b = mae + t
    np.savetxt(mydir+"b.csv", b, delimiter=',')
