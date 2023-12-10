import torch
import numpy as np 
from torch import nn
from torch.nn import functional as F
from scipy import io
from torchsummary import summary


def conv1x3(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1)

def avgpool(x):
    out = torch.mean(x, dim=2)
    out = out.unsqueeze(1)
    return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv1 = conv1x3(in_channels, out_channels)
        #self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = conv1x3(out_channels, out_channels)
        #self.bn2 = nn.BatchNorm1d(out_channels)

        self.extra = nn.Sequential()
        if out_channels != in_channels:
            self.extra = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
                #nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                #nn.BatchNorm1d(out_channels)

            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        #out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        #out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out

class Resone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resone, self).__init__()
        self.conv1 = conv1x3(in_channels, out_channels)
        #self.bn1 = nn.BatchNorm1d(out_channels)
        self.extra = nn.Sequential()
        if out_channels != in_channels:
            self.extra = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
                #nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                #nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.extra(x) + out
        return out


class FECNN(nn.Module):
    def __init__(self):
        super(FECNN, self).__init__()
        self.conv1 = conv1x3(1, 40)
        self.blk1 = Resone(40, 40)
        self.pool1 = conv1x3(40, 40, 2)
        self.blk2 = ResBlock(40, 40)
        self.pool2 = conv1x3(40, 40, 2)
        self.blk3 = ResBlock(40, 40)
        self.pool3 = conv1x3(40, 40, 2)
        self.blk4 = Resone(40, 40)
        self.pool4 = conv1x3(40, 40, 2)
        self.pool5 = nn.Conv1d(40,1,1)
        self.fc = nn.Linear(12,12)    # 189bands--12   224bands--14  175bands--11  188bands--12   191bands--12bands  204bands--13   126bands--8
        #self.pool5 = nn.AvgPool1d(kernel_size=12, stride=1, padding=0)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print('first conv:', x.shape)
        x = F.relu(self.blk1(x))
        x = F.relu(self.pool1(x))
        x = F.relu(self.blk2(x))
        x = F.relu(self.pool2(x))
        x = F.relu(self.blk3(x))
        x = F.relu(self.pool3(x))
        x = F.relu(self.blk4(x))
        x = F.relu(self.pool4(x))
        x = F.relu(self.pool5(x))
        x = x.flatten(1)
        x = self.fc(x)
        #x = x.flatten()
        #print('after conv:', x.shape)
        #     x = avgpool(x)
        #x = self.pool5(x)
        #x = x.permute(0,2,1)

        return x

'''
def main():
    tmp = torch.randn(40, 40, 189)
    blk = ResBlock(40, 40)
    out = blk(tmp)
    print('block:', out.shape)

    x = torch.randn(1,1,189)
    model = FECNN()
    out = model(x)
    print('FECNN:', out.shape)



if __name__ == '__main__':
    main()
'''

class TripletNet(nn.Module):
    def __init__(self, FECNN):
        super(TripletNet, self).__init__()
        self.embedding_net = FECNN

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)




