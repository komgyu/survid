import cv2
import os
import numpy as np
#import random
#import matplotlib.pyplot as plt
#import collections
import torch
import torchvision
from torchvision import transforms
#import cv2
from PIL import Image
import torchvision.transforms.functional as TF
#import torchvision.transforms as transforms

# def add_Gaussian(flow):
#     flow = flow/255
#     row,col,ch= np.shape(flow)
#     mean = 0
#     sigma = 0.2
#     gauss = np.random.normal(mean,sigma,(row,col,ch))
#     gauss = gauss.reshape(row,col,ch)
#     noisy = flow + gauss
#     noisy = np.clip(noisy, 0, 1)
#     noisy = np.uint8(noisy*255)
#     return noisy

class TrainDataSet(torch.utils.data.Dataset):
    def __init__(self, root, list_path, ignore_label=255,device ='cuda'):
        super(TrainDataSet,self).__init__()
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.root, "original_data/scale/original/%s.png" % name)
            label_file = os.path.join(self.root, "original_data/scale/label/%s.png" % name)
            flow_file = os.path.join(self.root, "original_data/scale/flow/%s.png" % name)
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
        m = np.shape(label)[0]
        n = np.shape(label)[1]
        new_label = np.empty([m,n],dtype = np.uint8)
        for i in range(m):
            for j in range(n):
                if label[i, j, 0] ==128: 
                    new_label[i,j] = 2 #object red
                if label[i, j, 1] ==128:
                    new_label[i,j] = 1 # tool green
                if label[i, j, 1]!=128 and label[i, j, 0]!= 128:
                    new_label[i,j] = 0 #backgrond black
       
     
        size_origin = image.size # W * H
        I = np.asarray(np.array(image),np.float32) 
        I = I.transpose((2,0,1))#transpose the  H*W*C to C*H*W
        I = torch.tensor(I)
        F = np.asarray(np.array(flow),np.float32) 
        # F = add_Gaussian(F)
        F = F.transpose((2,0,1))#transpose the  H*W*C to C*H*W
        F = torch.tensor(F)
        IF = torch.cat((I,F), 0)
        # L = np.asarray(np.array(label),np.float32)
        # L = L.transpose((2,0,1))
        L = torch.tensor(new_label,dtype=torch.long)

        #print(I.shape,L.shape)
        return I, IF, L, np.array(size_origin), name

class AugDataSet(torch.utils.data.Dataset):
    def __init__(self, root, list_path, ignore_label=255,device ='cuda'):
        super(AugDataSet,self).__init__()
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.root, "original_data/scale/original/%s.png" % name)
            label_file = os.path.join(self.root, "original_data/scale/label/%s.png" % name)
            flow_file = os.path.join(self.root, "original_data/scale/flow/%s.png" % name)
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
        #transforms.RandomCrop(300,pad_if_needed=True,fill=0, padding_mode='constant')
        flip = transforms.RandomHorizontalFlip(p=1)
        name  = datafiles["name"]
        image = Image.open(datafiles["img"]).convert('RGB')
        image = flip(image)
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
        image = TF.crop(image, i, j, h, w)
        flow  = Image.open(datafiles["flow"]).convert('RGB')
        flow  = flip(flow)
        flow  = TF.crop(flow,  i, j, h, w)
        label = cv2.imread(datafiles["label"])# h,w,c [1024, 1920, 3]
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = Image.fromarray(label)
        label  = flip(label)
        label = TF.crop(label, i, j, h, w)

        label = np.asarray(label)
        m = np.shape(label)[0]
        n = np.shape(label)[1]
        new_label = np.empty([m,n],dtype = np.uint8)
        for i in range(m):
            for j in range(n):
                if label[i, j, 0]!= 0: 
                    new_label[i,j] = 2 #object red
                if label[i, j, 1] !=0 :
                    new_label[i,j] = 1 # tool green
                if label[i, j, 1]==0 and label[i, j, 0]== 0:
                    new_label[i,j] = 0 #backgrond black
       
     
        size_origin = image.size # W * H
        I = np.asarray(np.array(image),np.float32) 
        I = I.transpose((2,0,1))#transpose the  H*W*C to C*H*W
        I = torch.tensor(I)
        F = np.asarray(np.array(flow),np.float32) 
        # F = add_Gaussian(F)
        F = F.transpose((2,0,1))#transpose the  H*W*C to C*H*W
        F = torch.tensor(F)
        IF = torch.cat((I,F), 0)
        # L = np.asarray(np.array(label),np.float32)
        # L = L.transpose((2,0,1))
        L = torch.tensor(new_label,dtype=torch.long)

        #print(I.shape,L.shape)
        return I, IF, L, np.array(size_origin), name


class ValDataSet(torch.utils.data.Dataset):
    def __init__(self, root, list_path, ignore_label=255,device ='cuda'):
        super(ValDataSet,self).__init__()
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.root, "validate/scale/original/%s.png" % name)
            label_file = os.path.join(self.root, "validate/scale/label/%s.png" % name)
            flow_file = os.path.join(self.root, "validate/scale/flow/%s.png" % name)
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
        m = np.shape(label)[0]
        n = np.shape(label)[1]
        new_label = np.empty([m,n],dtype = np.uint8)
        for i in range(m):
            for j in range(n):
                if label[i, j, 0]== 128: 
                    new_label[i,j] = 2 #object red
                if label[i, j, 1] ==128:
                    new_label[i,j] = 1 # tool green
                if label[i, j, 1]!=128 and label[i, j, 0]!= 128:
                    new_label[i,j] = 0 #backgrond black
       
     
        size_origin = image.size # W * H
        I = np.asarray(np.array(image),np.float32) 
        I = I.transpose((2,0,1))#transpose the  H*W*C to C*H*W
        I = torch.tensor(I)
        F = np.asarray(np.array(flow),np.float32) 
        # F = add_Gaussian(F)
        F = F.transpose((2,0,1))#transpose the  H*W*C to C*H*W
        F = torch.tensor(F)
        IF = torch.cat((I,F), 0)
        # L = np.asarray(np.array(label),np.float32)
        # L = L.transpose((2,0,1))
        L = torch.tensor(new_label,dtype=torch.long)

        #print(I.shape,L.shape)
        return I, IF, L, np.array(size_origin), name
