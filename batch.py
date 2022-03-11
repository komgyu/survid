import cv2
import os
import numpy as np
#import random
#import matplotlib.pyplot as plt
#import collections
import torch
import torchvision
#import cv2
from PIL import Image
#import torchvision.transforms as transforms

class UAVDataSet(torch.utils.data.Dataset):
    def __init__(self, root, list_path, ignore_label=255,device ='cuda'):
        super(UAVDataSet,self).__init__()
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.root, "original_data/original/%s.png" % name)
            label_file = os.path.join(self.root, "original_data/label/%s.png" % name)
            flow_file = os.path.join(self.root, "original_data/flow/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "flow": flow_file,
                "name": name
            })
    def __len__(self):
        return len(self.files)
 
 
    def __getitem__(self, index):
        datafiles = self.files[index]
 
        '''load the datas'''
        name  = datafiles["name"]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"]).convert('RGB')
        flow  = Image.open(datafiles["flow"]).convert('RGB')
        # label = cv2.imread(datafiles["label"])# h,w,c
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        size_origin = image.size # W * H
        I = np.asarray(np.array(image),np.float32) 
        I = I.transpose((2,0,1))#transpose the  H*W*C to C*H*W
        I = torch.tensor(I)
        F = np.asarray(np.array(flow),np.float32) 
        F = F.transpose((2,0,1))#transpose the  H*W*C to C*H*W
        F = torch.tensor(F)
        IF = torch.cat((I,F), 0)
        L = np.asarray(np.array(label),np.float32)
        L = L.transpose((2,0,1))
        L = torch.tensor(label)
        #print(I.shape,L.shape)
        return IF, L, np.array(size_origin), name




# if __name__ == '__main__':
#     DATA_DIRECTORY = './'
#     DATA_LIST_PATH = './images_id.txt'
#     Batch_size = 2
#     MEAN = (104.008, 116.669, 122.675)
#     dst = UAVDataSet(DATA_DIRECTORY,DATA_LIST_PATH)
#     # just for test,  so the mean is (0,0,0) to show the original images.
#     # But when we are training a model, the mean should have another value
#     trainloader = torch.utils.data.DataLoader(dst, batch_size = Batch_size)
#     plt.ion()
#     for i, data in enumerate(trainloader):
#         imgs, labels,_,_= data
#         if i % 1 == 0:
#             img = torchvision.utils.make_grid(imgs).numpy()
#             img = img.astype(np.uint8) # change the dtype from float32 to uint8, 
#                                        # because the plt.imshow() need the uint8
#             img = np.transpose(img, (1, 2, 0)) # transpose the C*H*W to H*W*C
#             #img = img[:, :, ::-1]
#             plt.imshow(img)
#             plt.show()
#             plt.pause(0.5)
 
# #            label = torchvision.utils.make_grid(labels).numpy()
#             labels = labels.numpy().astype(np.uint8) # change the dtype from float32 to uint8, 
# #                                       # because the plt.imshow() need the uint8
#             for i in range(labels.shape[0]):
#                 plt.imshow(labels[i],cmap='gray')
#                 plt.show()
#                 plt.pause(9)
 
#             #input()
