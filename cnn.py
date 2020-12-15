"""
cnn archi
"""

import torch as tr
import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self):
        super( CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                                out_channels = 16, 
                                kernel_size = 3,
                                stride = (1, 2),
                                padding= 1)
        self.a = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=16,
                                out_channels = 32,
                                kernel_size = 3,
                                stride = 1,
                                padding= 1)
        self.pooling1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=32,
                                out_channels = 16,
                                kernel_size = 3,
                                stride = 1,
                                padding= 1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(in_channels=16,
                                out_channels = 2,
                                kernel_size = 5,
                                stride = 1,
                                padding= 2)

    def forward(self,
                x,  #shape (batch_size,3,  201, 401 );  height = 201, width = 401, 3 channels
                ):

        height, width = x.shape[-2], x.shape[-1]
        print("input", x.shape)

        x = self.a(self.conv1(x)) 
        print("conv1", x.shape)

        x = self.a(self.conv2(x))
        print("conv2", x.shape)

        x = self.pooling1(x)
        print("pool1", x.shape)

        x = self.a(self.conv3(x))
        print("conv3", x.shape)

        x = self.upsample(x)
        print("upsample", x.shape)

        x = self.conv4(x)
        print("conv4", x.shape)

        return(x.view(-1, 1, 201, 401))



if __name__ == "__main__":

    cnn = CNN()

    # input shape (batch_size, 201, 401, 3)

    fake_input = tr.rand(2, 3, 201, 401)


    print(cnn(fake_input).shape)