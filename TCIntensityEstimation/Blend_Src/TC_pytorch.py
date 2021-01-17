import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np
import argparse
from tools import normalize_data, rotate_by_channel, blend_loss

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2d_0 = nn.Conv2d(2, 16, 4, stride=2) # 64 in
        self.conv2d_1 = nn.Conv2d(16, 32, 3, stride=2, padding=1) # 32 in
        self.conv2d_2 = nn.Conv2d(32, 64, 3, stride=2, padding=1) # 16 in
        self.conv2d_3 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # 8 in 4 out
        self.linear_0 = nn.Linear(4*4*128, 256)
        self.linear_1 = nn.Linear(256, 64)
        self.linear_2 = nn.Linear(64, 1)

    def forward(self, train_img):
        x = F.relu(self.conv2d_0(train_img))
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.relu(self.conv2d_3(x))
    
        xb = x.contiguous().view(-1, 4*4*128)
        xb = F.relu(self.linear_0(xb))
        xb = F.relu(self.linear_1(xb))
        xb = self.linear_2(xb)
        return xb

class BlendNet(nn.Module):
    def __init__(self, device, max_rotated_sita, blend_num):
        super(BlendNet, self).__init__()
        self.net = Net().to(device)
        self.MAX_ROTATED_SITA = max_rotated_sita
        self.blend_num = blend_num
        self.device = device
        self.transforms = transforms.RandomRotation(self.MAX_ROTATED_SITA, PIL.Image.BILINEAR)

    def forward(self, train_img):
        pred = torch.zeros((self.blend_num, train_img.shape[0])).to(self.device)
        for i in range(self.blend_num):
            with torch.no_grad():
                train_x = self.transforms(train_img)
                #print(f'transformed train_img: {train_img.shape}')
                train_x = train_x[:, 18:82, 18:82, :].permute(0, 3, 1, 2)
            pred[i] = self.net(train_x).squeeze(-1)
        return pred


def train(args, model, optimizer, scheduler):
    x_train = np.load(args.trainset_xpath).astype('float32')
    y_train = np.load(args.trainset_ypath).astype('float32')
    train_img_num = x_train.shape[0]
    arr = np.arange(train_img_num)
    device = args.device
    BATCH_SIZE = args.batch_size
    model.train()
    for epoch in range(args.epochs):
        np.random.shuffle(arr)
        #epoch_loss = torch.zeros(train_img_num//BATCH_SIZE+1)
        loss_sum = 0
        for batch in range(0, train_img_num, BATCH_SIZE):
            batch_x = torch.tensor(x_train[arr[batch:batch+BATCH_SIZE]]).to(device)
            batch_y = torch.tensor(y_train[arr[batch:batch+BATCH_SIZE]]).to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            #print(type(output), type(batch_y))
            mseloss, varloss = blend_loss(output, batch_y, alpha=0.5)
            #epoch_loss[batch // BATCH_SIZE] = loss
            loss = 0.5 * mseloss + (1 - 0.5) * varloss
            loss_sum += loss
            loss.backward()
            optimizer.step()
        if epoch % args.log_interval == 0:
            print(f'Train Epoch: {epoch}, mseloss: {mseloss//1}, varloss: {varloss//1}, blend_loss: {loss_sum/(train_img_num//BATCH_SIZE)//1}')
        scheduler.step()

def test(args):        
    x_test  = np.load(args.testset_xpath).astype('float32')
    y_test  = np.load(args.testset_ypath).astype('float32')
    x_test = x_test[y_test<=180,:,:,:]
    y_test = y_test[y_test<=180]

def main(args):
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args.save_model)
    model = BlendNet(args.device, max_rotated_sita=180, blend_num=4).to(args.device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)
    train(args, model, optimizer, scheduler)
    # TO DO
    # Save the trained Model
    if args.save_model:
        torch.save(model.state_dict(), "epoch_"+str(args.epochs)+".pt")
    # Test & Evaluation


if __name__=="__main__":
    parser =  argparse.ArgumentParser()
    parser.add_argument("-B", "--batch_size", default=64)
    
    parser.add_argument("-P", "--datapath", default="../Data/TCIR-ATLN_EPAC_WPAC.h5", help="the TCIR dataset file path")
    parser.add_argument("-Tx", "--trainset_xpath", default="../Data/ATLN_2003_2014_data_x_101.npy", help="the trainning set x file path")
    parser.add_argument("-Ty", "--trainset_ypath", default="../Data/ATLN_2003_2014_data_y_101.npy", help="the trainning set y file path")
    parser.add_argument("-Tex", "--testset_xpath", default="../Data/ATLN_2015_2016_data_x_101.npy", help="the test set x file path")
    parser.add_argument("-Tey", "--testset_ypath", default="../Data/ATLN_2015_2016_data_y_101.npy", help="the test set y file path")
    parser.add_argument("-E", "--epochs", default=300, help="epochs for trainning")
    parser.add_argument("--log_interval", default=1)
    parser.add_argument("--save_model", action='store_false')

    args = parser.parse_args()
    args.lr = 1e-3
    args.gamma = 0.90
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
