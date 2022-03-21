import os
import argparse
import time
import pickle
import torch
import torchvision.utils as vutils
import numpy as np
from PIL import Image

from snet import SNet
from batch import UAVDataSet
from loss import DiceLoss


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

    # load the data
    with open(args.data_dir + '/data/data.pck', 'rb') as f:
        data = pickle.load(f)
    x = data['x']
    f = data['f']
    #xf = torch.cat((x,f), 1).to(device)
    xf = data['x'].to(device)
    s_gt = data['s'].to(device)
    print('# Dataset loaded', file=open(logname, 'a'), flush=True)

    #######################################################################
    DATA_DIRECTORY = './'
    DATA_LIST_PATH = './images_id.txt'
    Batch_size = 1
    # MEAN = (104.008, 116.669, 122.675)
    dst = UAVDataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    # just for test,  so the mean is (0,0,0) to show the original images.
    # But when we are training a model, the mean should have another value
    trainloader = torch.utils.data.DataLoader(dst, batch_size = Batch_size, shuffle=True)
    #########################################################################

    log_period = 10
    save_period = 100
    count = 0

    print('# Everything prepared, go ...', file=open(logname, 'a'), flush=True)

    loss = 0.69*torch.ones([], device=device) # loss for random decision = ln(0.5)
    accuracy = 0.5*torch.ones([], device=device) # random decision = 0.5
    
#    
    while True:
     for i, data in enumerate(trainloader):
        s, s_gt,_,_= data
        s    = s.to(device)
        s_gt = s_gt.to(device)

        optimizer.zero_grad()
        scores = snet(s) 
    
        print(s_gt.size())  #[1,1024,1920]
        print(scores.size()) #[1, 3, 1024, 1920]


        current_loss = criterion(scores, s_gt)
        
        current_loss.backward()
        optimizer.step()
        # track the loss
        loss = loss*0.99 + current_loss.detach()*0.01

        # track the accuracy
        #s = (scores.detach()>0).float() # current decision
        #decide s
        
        soft = torch.nn.Softmax(dim = 1)
        s = soft(scores)
        print(s.size()) #[1,3,1024,1920]
        s = torch.argmax(s, dim = 1)

        current_accuracy = (s == s_gt).float().mean()
        accuracy = accuracy*0.99 + current_accuracy*0.01

        # once awhile print something out
        if count % log_period == log_period-1:
            strtoprint = 'time: ' + str(time.time()-time0) + ' count: ' + str(count)

            strtoprint += ' len: ' + str(vlen(snet).cpu().numpy())
            strtoprint += ' loss: ' + str(loss.cpu().numpy())
            strtoprint += ' accuracy: ' + str(accuracy.cpu().numpy())

            print(strtoprint, file=open(logname, 'a'), flush=True)

        # once awhile save the models for further use
        if count % save_period == 0:
            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            torch.save(snet.state_dict(), args.data_dir + '/models/snet_' + args.call_prefix + '.pt')

            # visualize current result (for one image)
            
            imtoviz = s
            imtoviz = imtoviz.cpu().detach().numpy()
            s_gt = s_gt.cpu().detach().numpy()
 
            g_viz = np.zeros([1024,1920,3],dtype = np.uint8)
            for i in range(1024):
                for j in range(1920):
                    if s_gt[0,i,j] ==1: 
                        g_viz[i,j,1]= 128
                    if s_gt[0,i,j] ==2:
                        g_viz[i,j,0]=  127 
                    if s_gt[0,i,j] ==0:
                        g_viz[i,j,:] = 0                   
            g_viz = Image.fromarray(g_viz)
            
             
            im_viz = np.zeros([1024,1920,3])
            for i in range(1024):
                for j in range(1920):
                    if imtoviz[0,i,j] ==1:
                        im_viz[i,j,1]= 128
                        
                    if imtoviz[0,i,j] ==2:
                        im_viz[i,j,0]=  128
                    if imtoviz[0,i,j] ==0 :
                        im_viz[i,j,:]=  0
            im_viz = Image.fromarray(np.uint8(im_viz))

            g_viz = g_viz.save('./images/Cross/gt.png')
            im_viz = im_viz.save('./images/Cross/img_' + str(count) + '.png')
            #vutils.save_image(g_viz.float(), args.data_dir + '/images/Cross/gt.png')
            #vutils.save_image(im_viz.float(), args.data_dir + '/images/Cross/img_' + str(count) + '.png')
            print('# Now at ' + time.strftime('%c'), file=open(logname, 'a'), flush=True)
            print('# ... done.', file=open(logname, 'a'), flush=True)

        count += 1
        if count == args.niterations:
            break
        torch.cuda.empty_cache()
    
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

