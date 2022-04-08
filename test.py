import torch
import pickle
import numpy as np
import torchvision.utils as vutils

class SNet(torch.nn.Module):
    def __init__(self, nch):
        super(SNet, self).__init__()

        nn = 50
        wnd = 5
        # self.conv0 = torch.nn.Conv2d(nch, nn, wnd, padding='same')  #input channel, output channel, wnd kernel size 
        self.pool0 = torch.nn.MaxPool2d(2, 2)
        # self.conv1 = torch.nn.Conv2d(nn,  nn, wnd, padding='same')  #the output size is the same as the input size(when stride=1).
        # self.pool1 = torch.nn.MaxPool2d(2, 2)
        # self.conv2 = torch.nn.Conv2d(nn,  nn, wnd, padding='same')
        # self.pool2 = torch.nn.MaxPool2d(2, 2)
        # self.conv3 = torch.nn.Conv2d(nn,  1,  wnd, padding='same')
        self.conv0 = torch.nn.Conv2d(nch, nn, wnd, padding='same') #nch:number of input channel
        self.conv1 = torch.nn.Conv2d(nn,  nn, wnd, padding='same')
        self.conv2 = torch.nn.Conv2d(nn,  nn, wnd, padding='same')
        self.conv3 = torch.nn.Conv2d(nn,  nn, wnd, padding='same')
        self.conv4 = torch.nn.Conv2d(nn,  3,  wnd, padding='same')
    def forward(self, x):

        # hh = self.conv0(x)
        # hh = self.pool0(torch.relu(self.conv0(hh)))
        # hh = self.conv1(hh)
        # hh = self.pool1(torch.relu(self.conv1(hh)))
        # return hh
        hh = torch.relu(self.conv0(x))
        hh = self.pool0(torch.relu(hh))
        hh = torch.relu(self.conv1(hh))
        hh = torch.relu(self.conv2(hh))
        return self.conv3(hh)

print(torch.cuda.is_available())

device = torch.cuda.current_device()

with open('D:/yukong/survid' + '/data/data.pck', 'rb') as f:
    data = pickle.load(f)
    x = data['x']
    f = data['f']
    xf = torch.cat((x,f), 1).to(device)
    s_gt = data['s'].to(device)

snet = torch.nn.DataParallel(SNet(6)).to(device)

print(xf.size())
scores = snet(xf)
s = (scores.detach()>0).float()

print(s.size())
print(s_gt.size())

# s_array = np.asarray(s.cpu())

vutils.save_image(s[0], './images/test.png')
    # i = np.asarray(i.cpu())
    # print(np.shape(i))
    # img = Image.fromarray(i, 'RGB')
    # img.save('my.png')
    # img.show()