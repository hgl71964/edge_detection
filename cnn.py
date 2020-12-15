"""
cnn archi
"""

import torch as tr
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super( CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                                out_channels = 16, 
                                kernel_size = (),
                                stride = 1,
                                padding= 1)

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
                x,  #shape (batch_size, 201, 401, 3);  height = 201, width = 401, 3 channels
                ):

        x = nn.LeakyReLU(self.conv1(x)) 
        print(x.shape)

        x = self.pooling1(x)
        print(x.shape)

        x = nn.LeakyReLU(self.conv3(x))
        print(x.shape)

        x = self.upsample(x)
        print(x.shape)

        x = self.conv4(x)
        print(x.shape)

        return(x.view(-1, 201, 401, 1))



if __name__ == "__main__":

    cnn = CNN()

    # input shape (batch_size, 201, 401, 3)

    fake_input = tr.rand(2, 201, 401, 3)


    print(cnn(fake_input).shape)