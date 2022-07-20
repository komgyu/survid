import cv2
import time
import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image    
from snet import SNet       
def computeIoU(scores, labels):
    #scores [4,256,480]
    iou = 0
    m = scores.shape[0]
    x = scores.shape[1]
    y = scores.shape[2]
    for i in range(m):
        #scores   = scores.cpu().detach().numpy()
        #labels   = labels.cpu().detach().numpy()

        count = np.zeros([x,y],dtype = np.uint8)
        count = np.logical_and(labels[i,:,:]==0, scores[i,:,:]==0)
        num00 = np.count_nonzero(count)
    
        count = np.zeros([x,y],dtype = np.uint8)
        count = np.logical_and(labels[i,:,:]==0, scores[i,:,:]==1)
        num01 = np.count_nonzero(count)

        count = np.zeros([x,y],dtype = np.uint8)
        count = np.logical_and(labels[i,:,:]==0, scores[i,:,:]==2)
        num02 = np.count_nonzero(count)

        count = np.zeros([x,y],dtype = np.uint8)
        count = np.logical_and(labels[i,:,:]==1, scores[i,:,:]==0)
        num10 = np.count_nonzero(count)

        count = np.zeros([x,y],dtype = np.uint8)
        count = np.logical_and(labels[i,:,:]==1, scores[i,:,:]==1)
        num11 = np.count_nonzero(count)

        count = np.zeros([x,y],dtype = np.uint8)
        count = np.logical_and(labels[i,:,:]==1, scores[i,:,:]==2)
        num12 = np.count_nonzero(count)

        count = np.zeros([x,y],dtype = np.uint8)
        count = np.logical_and(labels[i,:,:]==2, scores[i,:,:]==0)
        num20 = np.count_nonzero(count)

        count = np.zeros([x,y],dtype = np.uint8)
        count = np.logical_and(labels[i,:,:]==2, scores[i,:,:]==1)
        num21 = np.count_nonzero(count)

        count = np.zeros([x,y],dtype = np.uint8)
        count = np.logical_and(labels[i,:,:]==2, scores[i,:,:]==2)
        num22 = np.count_nonzero(count)

        if num00 == 0:
            iou0 = 0
        else:
            iou0 = num00/(num00+num01+num02+num10+num20)
        
        if num11 ==0:
            iou1 =0
        else:
            iou1 = num11/(num10+num11+num12+num01+num21)

        if num22==0:
            iou2 = 0
        else:
            iou2 = num22/(num20+num21+num22+num02+num12)

        miou = (iou0 + iou1 + iou2)/3
        iou = iou+miou

    iou = iou/m
    return iou

# def add_Gaussian(flow):
#     flow = flow/255
#     row,col,ch= np.shape(flow)
#     mean = 0
#     sigma = 0.5
#     gauss = np.random.normal(mean,sigma,(row,col,ch))
#     gauss = gauss.reshape(row,col,ch)
#     noisy = flow + gauss
#     noisy = np.clip(noisy, 0, 1)
#     noisy = np.uint8(noisy*255)
#     return noisy

lst = [2, 95, 96, 97, 98, 99, 110, 111, 112, 113, 114, 500, 501, 502, 503, 504, 735, 736, 737, 738,

       739, 750, 751, 752, 753, 754, 900, 901, 902, 903, 904, 3146,  3861, 4364, 4411, 5382 ]

# lst = [300, 301, 302, 303, 304, 305, 306, 307, 308, 309]


for id in lst:

    IMAGE_PATH = './test/scale/original/'+str(id)+'.png'
    FLOW_PATH = './test/scale/flow/' +str(id)+'.png'
    # FLOW_PATH = 'dresden1.png'
    LABEL_PATH = './test/scale/label/'+ str(id)+'.png'
    PATH = './models/snet_tmp6final.pt'
    device = torch.cuda.current_device()
    model = torch.nn.DataParallel(SNet(6)).to(device)
    model.load_state_dict(torch.load(PATH))
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')


    image = Image.open(IMAGE_PATH).convert('RGB')
    image = np.asarray(np.array(image),np.float32) 
    image = image.transpose((2,0,1))
    image = torch.tensor(image)
    image = image.to(device)

    flow = Image.open(FLOW_PATH).convert('RGB')
    flow = np.asarray(np.array(flow),np.float32) 

    #flow = add_Gaussian(flow)

    m = np.shape(flow)[0]
    n = np.shape(flow)[1]
    flow = flow.transpose((2,0,1))#transpose the  H*W*C to C*H*W
    flow = torch.tensor(flow)
    flow = flow.to(device)

    label= cv2.imread(LABEL_PATH)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

    new_label = np.empty([m,n],dtype = np.uint8)
    for i in range(m):
      for j in range(n):
        if label[i, j, 0]!= 0: 
            new_label[i,j] = 2 #object red
        if label[i, j, 1] !=0:
            new_label[i,j] = 1 # tool green
        if label[i, j, 1]==0 and label[i, j, 0]== 0:
            new_label[i,j] = 0 #backgrond black

    L = torch.tensor(new_label,dtype=torch.long)
    L = L.unsqueeze(0)
    L = L.to(device)

    IF = torch.cat((image,flow), 0)
    IF = IF.unsqueeze(0)

    image = image.unsqueeze(0)

    scores = model(IF)

    scores = torch.argmax(scores, dim = 1)
    accuracy = (scores == L).float().mean()

    # print(scores.size())
    scores = scores.cpu().detach().numpy()
    L   = L.cpu().detach().numpy()
    miou = computeIoU(scores, L)

    im_viz = np.zeros([m,n,3])
    # print(np.shape(im_viz))

    for i in range(256):
        for j in range(480):
            if scores[0,i,j]==0:
                im_viz[i,j,0] = 255
                im_viz[i,j,1] = 255
                im_viz[i,j,2] = 255
               
                
            if scores[0,i,j] == 2:
                im_viz[i,j,0] = 255
                im_viz[i,j,1] = 100
                im_viz[i,j,2] = 100
              
            if scores[0,i,j] == 1:
                im_viz[i,j,0] = 100
                im_viz[i,j,1] = 255
                im_viz[i,j,2] = 100



    # im_viz[:,:,0] = (scores[0,:,:]==0)*240
    # im_viz[:,:,0] = (scores[0,:,:]==2)*90
    # im_viz[:,:,1] = (scores[0,:,:]==0)*240
    # im_viz[:,:,1] = (scores[0,:,:]==1)*90
    # im_viz[:,:,2] = (scores[0,:,:]==0)*240



    # im_viz = Image.fromarray(np.uint8(im_viz))
    # im_viz = im_viz.save('./test/'+str(id)+'.png')

    # background = Image.open(IMAGE_PATH)
    # overlay = Image.open('./test/'+str(id)+".png")

    # background = background.convert("RGBA")
    # overlay = overlay.convert("RGBA")

    # new_img = Image.blend(background, overlay, 0.4)
    # new_img.save("./test/overlay" +str(id)+".png","PNG")

    print(id)
    print(accuracy)
    print(miou)
    
    strtoprint = ''
    strtoprint += str(id) +' '
    strtoprint += str(accuracy.cpu().numpy())+ ' '
    strtoprint +=  str(miou)
    print(strtoprint, file=open('./test/record.txt', 'a'), flush=True)