import os
import cv2
import argparse
import time
import pickle
import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image

from snet import SNet
from batch import TrainDataSet
# from loss import DiceLoss


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def vlen(layer): #vlen(snet)
    vsum = 0.
    vnn = 0
    for vv in layer.parameters():
        if vv.requires_grad: # requires_grad = False default is False
            param = vv.data
            vsum = vsum + (param*param).sum()
            vnn = vnn + param.numel()  #返回param中元素的数量
    return vsum/vnn

# entry point, the main
def main():

    time0 = time.time()

    # printout
    logname = args.data_dir + '/logs/Cross/log-' + args.call_prefix + '.txt' #see 115
    print('# Starting at ' + time.strftime('%c'), file=open(logname, 'w'), flush=True)

    # we are using GPU
    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    # the network
    snet = torch.nn.DataParallel(SNet(6)).to(device) # 6 input channels (image+flow), multi-GPU
    print('# Using', torch.cuda.device_count(), 'GPUs', file=open(logname, 'a'), flush=True)

    # lerning related stuff
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    #criterion = DiceLoss()
    #criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(snet.parameters(), lr=args.stepsize)


    #######################################################################
    TRAIN_DATA_DIRECTORY = './'
    TRAIN_DATA_LIST_PATH = './images_id.txt'
    Batch_size = 1
    # MEAN = (104.008, 116.669, 122.675)
    dst = TrainDataSet(TRAIN_DATA_DIRECTORY,TRAIN_DATA_LIST_PATH)
    # just for test,  so the mean is (0,0,0) to show the original images.
    # But when we are training a model, the mean should have another value
    trainloader = torch.utils.data.DataLoader(dst, batch_size = Batch_size, shuffle=True)
    #########################################################################

    log_period = 1
    save_period = 50
    epoch = 0

    print('# Everything prepared, go ...', file=open(logname, 'a'), flush=True)

    train_loss = 0.69*torch.ones([], device=device) # loss for random decision = ln(0.5)
    train_accuracy = 0.5*torch.ones([], device=device) # random decision = 0.5

    val_loss = 0.69*torch.ones([], device=device) # loss for random decision = ln(0.5)
    val_accuracy = 0.5*torch.ones([], device=device) # random decision = 0.5
    
#    
    while epoch<= 5000:
        for i, data in enumerate(trainloader):
            im, s, s_gt,_,_= data #im，original image, s = im + flow
            im   = im.to(device)
            s    = s.to(device)
            s_gt = s_gt.to(device)

            optimizer.zero_grad()
            scores = snet(s) 

            current_loss = criterion(scores, s_gt)
            
            current_loss.backward()
            optimizer.step()
            # track the loss
            train_loss = train_loss*0.99 + current_loss.detach()*0.01
            s = torch.argmax(scores, dim = 1)

            current_accuracy = (s == s_gt).float().mean()
            train_accuracy = train_accuracy*0.99 + current_accuracy*0.01

        # once awhile print something out   
        if epoch%log_period==0:
            strtoprint = 'time: ' + str(time.time()-time0) + ' epoch: ' + str(epoch)
            strtoprint += ' len: ' + str(vlen(snet).cpu().numpy())
            strtoprint += ' train_loss: ' + str(train_loss.cpu().numpy())
            strtoprint += ' train_accuracy: ' + str(train_accuracy.cpu().numpy())
            print(strtoprint, file=open(logname, 'a'), flush=True)


            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            torch.save(snet.state_dict(), args.data_dir + '/models/snet_' + args.call_prefix + '.pt')

            # visualize current result (for one image)
            
            imtoviz = s
            imtoviz = imtoviz.cpu().detach().numpy()
            s_gt = s_gt.cpu().detach().numpy()

            g_viz = np.zeros([1024,1920,3],dtype = np.uint8)
            g_viz[:,:,0] = (s_gt[0,:,:]== 2)*128 
            g_viz[:,:,1] = (s_gt[0,:,:]== 1)*128                  
            g_viz = Image.fromarray(g_viz)
            
             
            im_viz = np.zeros([1024,1920,3])
            im_viz[:,:,0] = (imtoviz[0,:,:]==2)*128
            im_viz[:,:,1] = (imtoviz[0,:,:]==1)*128

            im_viz = Image.fromarray(np.uint8(im_viz))

            g_viz = g_viz.save('./images/Cross/gt.png')
            im_viz = im_viz.save('./images/Cross/img_' + str(epoch) + '.png')
            #vutils.save_image(g_viz.float(), args.data_dir + '/images/Cross/gt.png')
            #vutils.save_image(im_viz.float(), args.data_dir + '/images/Cross/img_' + str(count) + '.png')
            print('# Now at ' + time.strftime('%c'), file=open(logname, 'a'), flush=True)
            print('# ... done.', file=open(logname, 'a'), flush=True)

            ####################################################################
            IMAGE_PATH = './validate/2peg/original/'+ str(300+epoch%9)+'.png'
            FLOW_PATH = './validate/2peg/flow/'+ str(300+epoch%9)+'.png'
            LABEL_PATH = './validate/2peg/label/'+ str(300+epoch%9)+'.png'
            PATH = 'D:/yukong/survid/models/snet_tmp.pt'
            model = torch.nn.DataParallel(SNet(6)).to(device)
            model.load_state_dict(torch.load(PATH))

            image = Image.open(IMAGE_PATH).convert('RGB')
            image = np.asarray(np.array(image),np.float32) 
            image = image.transpose((2,0,1))
            image = torch.tensor(image)
            image = image.to(device)

            flow = Image.open(FLOW_PATH).convert('RGB')
            flow = np.asarray(np.array(flow),np.float32) 
            flow = flow.transpose((2,0,1))#transpose the  H*W*C to C*H*W
            flow = torch.tensor(flow)
            flow = flow.to(device)

            label= cv2.imread(LABEL_PATH)
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
            
            L = torch.tensor(new_label,dtype=torch.long)
            L = L.unsqueeze(0)
            L = L.to(device)
            
            IF = torch.cat((image,flow), 0)
            IF = IF.unsqueeze(0)
            scores = model(IF)

            current_loss = criterion(scores, L)
            val_loss = val_loss*0.99 + current_loss.detach()*0.01
            
            scores = torch.argmax(scores, dim = 1)
            current_accuracy = (scores == L).float().mean()
            val_accuracy = val_accuracy*0.99 + current_accuracy*0.01

           

            predicttoprint = 'time: ' + str(time.time()-time0) + ' epoch: ' + str(epoch)
            predicttoprint += ' len: ' + str(vlen(snet).cpu().numpy())
            predicttoprint += ' val_loss: ' + str(val_loss.cpu().numpy())
            predicttoprint += ' val_accuracy: ' + str(val_accuracy.cpu().numpy())
            print(predicttoprint, file=open('D:/yukong/survid/logs/Cross/val_log.txt', 'a'), flush=True)
            print('# Finished ...', file=open('D:/yukong/survid/logs/Cross/val_log.txt', 'a'), flush=True)

        epoch += 1
        print(epoch)
        print(train_accuracy)
        print(val_accuracy)
        torch.cuda.empty_cache()

        if epoch == args.niterations:
            break
        
    
    print('# Finished at ' + time.strftime('%c') + ', %g seconds elapsed' %
          (time.time()-time0), file=open(logname, 'a'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', default='.', help='Data directory.')
    parser.add_argument('--call_prefix', default='tmp', help='Call prefix.')
    parser.add_argument('--stepsize', type=float, default=1e-4, help='Gradient step size.')
    parser.add_argument('--niterations', type=int, default=-1, help='The total numer of iterations (-1 for infinity)')

    args = parser.parse_args() 

    ensure_dir(args.data_dir + '/logs')
    ensure_dir(args.data_dir + '/models')
    ensure_dir(args.data_dir + '/images')

    main()
