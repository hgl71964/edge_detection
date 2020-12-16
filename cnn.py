import torch as tr
import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
                                padding= (1,1))

        self.upsample = nn.Upsample(scale_factor=(2,4),
                                    )

        self.conv4 = nn.Conv2d(in_channels=16,
                                out_channels = 1,
                                kernel_size = 2,
                                stride = 1,
                                padding= (1, 1))

    def forward(self,
                x,  #shape (batch_size, 3,  201, 401 );  3 channels, height = 201, width = 401, 
                verbose = False,
                ):

        height, width = x.shape[-2], x.shape[-1]
        if verbose:
            print("input", x.shape)

        x = self.a(self.conv1(x)) 
        if verbose:
            print("conv1", x.shape)

        x = self.a(self.conv2(x))
        if verbose:
            print("conv2", x.shape)

        x = self.pooling1(x)
        if verbose:
            print("pool1", x.shape)

        x = self.a(self.conv3(x))
        if verbose:
            print("conv3", x.shape)

        x = self.upsample(x)
        if verbose:
            print("upsample", x.shape)


        x = self.conv4(x)
        if verbose:
            print("conv4", x.shape)

        return(x.view(-1, 1, height, width))



if __name__ == "__main__":

    cnn = CNN()

    # input shape (batch_size, 201, 401, 3)

    fake_input = tr.rand(2, 3, 201, 401)


    print(cnn(fake_input).shape)