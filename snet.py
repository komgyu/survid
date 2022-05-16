import torch

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channel, out_channel,kernel_size):
        super(DoubleConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 3, padding = 1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channel, out_channel, 3, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU()
        )

    def forward(self, input):
        return self.conv(input)

class SNet(torch.nn.Module):
    def __init__(self, nch):
        super(SNet, self).__init__()
        self.conv1 = DoubleConv(nch, 32, 3)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64, 3)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128, 3)
        self.pool3 = torch.nn.AvgPool2d(2)
        self.conv4 = DoubleConv(128, 256 , 3)
        self.pool4 = torch.nn.AvgPool2d(2)
        self.conv5 = DoubleConv(256, 512, 3)
        self.up1   = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256, 3)
        self.up2   = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128, 3)
        self.up3   = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64, 3)
        self.up4   = torch.nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(32, 3, 3)


    def forward(self, x):
        hh1    = self.conv1(x)
        pool1 = self.pool1(hh1)
        hh2   = self.conv2(pool1)
        pool2 = self.pool2(hh2)
        hh3   = self.conv3(pool2)
        pool3 = self.pool3(hh3)
        hh4   = self.conv4(pool3)
        pool4 = self.pool4(hh4)
        hh5   = self.conv5(pool4)
        up1   = self.up1(hh5)
        cat1  = torch.cat([up1, hh4],dim = 1) # 128+128
        hh6   = self.conv6(cat1)
        up2   = self.up2(hh6)
        cat2  = torch.cat([up2, hh3], dim = 1) #64 + 64
        hh7   = self.conv7(cat2)
        up3   = self.up3(hh7)
        cat3  = torch.cat([up3, hh2], dim = 1) #32 + 32
        hh8   = self.conv8(cat3)
        up4   = self.up4(hh8)  
        cat4  = torch.cat([up4, hh1], dim = 1) #16 + 16
        hh9   = self.conv9(up4)

        return(hh9)
