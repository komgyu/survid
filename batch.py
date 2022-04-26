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
            img_file = os.path.join(self.root, "original_data/original/scale/%s.png" % name)
            label_file = os.path.join(self.root, "original_data/label/scale/%s.png" % name)
            flow_file = os.path.join(self.root, "original_data/flow/scale/%s.png" % name)
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
        flow  = Image.open(datafiles["flow"]).convert('RGB')
        label = cv2.imread(datafiles["label"])# h,w,c [1024, 1920, 3]
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        new_label = np.empty([1024,1920],dtype = np.uint8)
        for i in range(1024):
        	for j in range(1920):
        		if label[i, j, 0]!= 0: 
        			new_label[i,j] = 2 #object red
        		if label[i, j, 1] !=0:
        			new_label[i,j] = 1 # tool green
        		if label[i, j, 1]==0 and label[i, j, 0]== 0:
        			new_label[i,j] = 0 #backgrond black
       
     
        size_origin = image.size # W * H
        I = np.asarray(np.array(image),np.float32) 
        I = I.transpose((2,0,1))#transpose the  H*W*C to C*H*W
        I = torch.tensor(I)
        F = np.asarray(np.array(flow),np.float32) 
        F = F.transpose((2,0,1))#transpose the  H*W*C to C*H*W
        F = torch.tensor(F)
        IF = torch.cat((I,F), 0)
        # L = np.asarray(np.array(label),np.float32)
        # L = L.transpose((2,0,1))
        L = torch.tensor(new_label,dtype=torch.long)

        #print(I.shape,L.shape)
        return I, IF, L, np.array(size_origin), name


