
# import os
# from utils.dataset import alldataDataset
# from torch.utils.data import DataLoader

from torch import nn as nn
import torch
from torch.nn.modules.activation import Softmax, Softmax2d

class block1(nn.Module):
    def __init__(self):
        super(block1, self).__init__()
        self.con1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,dilation=1,padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,dilation=1,padding=(1,1)),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x=self.con1(x)
        return x

class block2(nn.Module):
    def __init__(self):
        super(block2,self).__init__()
        self.con2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,dilation=2,padding=(2,2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,dilation=2,padding=(2,2)),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x=self.con2(x)
        return x


class block3(nn.Module):
    def __init__(self):
        super(block3, self).__init__()
        self.con3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,dilation=4,padding=(4,4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,dilation=4,padding=(4,4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=4, padding=(4, 4)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.con3(x)
        return x

class block4(nn.Module):
    def __init__(self):
        super(block4, self).__init__()
        self.con4 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,dilation=8,padding=(8,8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,dilation=8,padding=(8,8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=8, padding=(8, 8)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.con4(x)
        return x

class block5(nn.Module):
    def __init__(self):
        super(block5, self).__init__()
        self.con5 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,dilation=16,padding=(16,16)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,dilation=16,padding=(16,16)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=16, padding=(16, 16)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.con5(x)
        return x


class block6(nn.Module):
    def __init__(self):
        super(block6,self).__init__()
        # self.ct=torch.cat((x1,x2,x3,x4,x5),dim=1)
        self.con6=nn.Sequential(
            nn.Conv2d(in_channels=320,out_channels=128,kernel_size=1),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=2,kernel_size=1),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            #nn.Softmax(dim=1)
        )
    def forward(self,x1,x2,x3,x4,x5):
        # print(x1.shape)
        # 这里前面使用了padding来保证卷积不改变图片尺寸，便于C通道concat
        ct=torch.cat((x1,x2,x3,x4,x5),dim=1)
        x=self.con6(ct)
        return x

# 测试各模块-->维度
# def test_train(net,devie):
#     fb_set = alldataDataset(root_dir='alldata_npy/')
#     fb_loader = DataLoader(dataset=fb_set, batch_size=1, shuffle=True)
#     net.train()
#
#     for data in fb_loader:
#         image = data['image'].to(device=device, dtype=torch.float)
#         label = data['mask'].to(device=device, dtype=torch.float)
#
#         pred = net(image)
#
#         print(pred.shape)
#         break
#
# if __name__=="__main__":
#     device=torch.device('cuda:0')
#     net=block3()
#     net=net.cuda()
#     net.to(device=device)
#     test_train(net,device)