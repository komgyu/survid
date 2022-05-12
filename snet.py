import torch

class SNet(torch.nn.Module):
    def __init__(self, nch):
        super(SNet, self).__init__()

        nn = 64
        wnd = 7
        self.conv0 = torch.nn.Conv2d(nch, 128, wnd, padding='same')
        self.conv1 = torch.nn.Conv2d(128,  128, wnd, padding='same') 
        self.conv2 = torch.nn.Conv2d(128,  128, wnd, padding='same')
        self.bn1   = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128,  128,  wnd, padding='same')
        self.pool1  = torch.nn.MaxPool2d(2)
        self.conv4 = torch.nn.Conv2d(128,  128,  wnd, padding='same')
        self.bn2   = torch.nn.BatchNorm2d(128)
        self.conv5 = torch.nn.Conv2d(128,  128,  wnd, padding='same')
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv6 = torch.nn.Conv2d(128,  128,  wnd, padding='same')
        self.up1   = torch.nn.ConvTranspose2d(128,128,2, stride = 2)
        self.conv7 = torch.nn.Conv2d(256,  256,  wnd, padding='same')
        self.up2   = torch.nn.ConvTranspose2d(256,128,2, stride = 2)
        self.conv8 = torch.nn.Conv2d(256,  3,  wnd, padding='same')
        
        # self.conv7 = torch.nn.Conv2d(64,  3,  wnd, padding='same')
      


    def forward(self, x):

        hh = torch.relu(self.conv0(x))
        hh = torch.relu(self.conv1(hh))
        hh = self.conv2(hh)
        hh = torch.relu(self.bn1(hh))
        hh0 = torch.relu(self.conv3(hh))   
        hh = self.conv4(self.pool1(hh0))
        hh = torch.relu(self.bn2(hh))
        hh1 = torch.relu(self.conv5(hh))
        hh = self.conv6(self.pool2(hh))
        hh = self.up1(hh)
        hh = torch.cat([hh,hh1], dim = 1)
        hh = torch.relu(self.conv7(hh))
        hh = self.up2(hh)
        hh = torch.cat([hh,hh0],dim = 1)
        # hh = torch.relu(self.conv8(hh))

        return self.conv8(hh)
