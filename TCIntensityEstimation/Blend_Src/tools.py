import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

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

def blend_loss(pred, gt, alpha=1.0):
    # pred: blend_num, batch_size
    # gt: batch_size
    mean = torch.mean(pred, axis=0)
    var = torch.var(pred, axis=0)
    #print(mean.device, gt.device)
    mse_loss = F.mse_loss(mean, gt)
    #loss = alpha * mse_loss + ((1 - alpha) * var.mean()) 
    return mse_loss, var.mean()
