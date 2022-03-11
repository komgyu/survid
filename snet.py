import torch

class SNet(torch.nn.Module):
    def __init__(self, nch):
        super(SNet, self).__init__()

        nn = 64
        wnd = 5
        self.conv0 = torch.nn.Conv2d(nch, 128, wnd, padding='same')
        self.conv1 = torch.nn.Conv2d(128,  128, wnd, padding='same') 
        self.conv2 = torch.nn.Conv2d(128,  128, wnd, padding='same')
        self.bn1   = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128,  128,  wnd, padding='same')
        self.conv4 = torch.nn.Conv2d(128,  128,  wnd, padding='same')
        #self.drop1  = torch.nn.Dropout(p=0.2) 
        self.bn2   = torch.nn.BatchNorm2d(128)
        self.conv5 = torch.nn.Conv2d(128,  64,  wnd, padding='same')
        self.conv6 = torch.nn.Conv2d(64,  3,  wnd, padding='same')
        # self.conv7 = torch.nn.Conv2d(64,  3,  wnd, padding='same')
      


    def forward(self, x):

        hh = torch.relu(self.conv0(x))
        hh = torch.relu(self.conv1(hh))
        hh = torch.relu(self.conv2(hh))
        hh = self.bn1(hh)
        hh = torch.relu(self.conv3(hh))
        hh = torch.relu(self.conv4(hh))
        hh = self.bn2(hh)
        hh = torch.relu(self.conv5(hh))
        # hh = torch.relu(self.conv6(hh))
       
      
        return self.conv6(hh)
