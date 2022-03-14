import torch
from snet import SNet
from PIL import Image
import numpy as np

IMAGE_PATH = 'C:/Users/yukong/Desktop/ground truth/3Gal/frame_0130.png'
FLOW_PATH = 'C:/Users/yukong/Desktop/ground truth/3Gal/130.png'
PATH = 'D:/yukong/survid/models/snet_tmp.pt'
device = torch.cuda.current_device()

model = torch.nn.DataParallel(SNet(6)).to(device)
model.load_state_dict(torch.load(PATH))

image = Image.open(IMAGE_PATH).convert('RGB')
image = np.asarray(np.array(image),np.float32) 
image = image.transpose((2,0,1))
image = torch.tensor(image)

flow = Image.open(FLOW_PATH).convert('RGB')
flow = np.asarray(np.array(flow),np.float32) 
flow = flow.transpose((2,0,1))#transpose the  H*W*C to C*H*W
flow = torch.tensor(flow)

IF = torch.cat((image,flow), 0)
IF = IF.unsqueeze(0)
scores = model(IF)

soft = torch.nn.Softmax(dim = 1)
out = soft(scores)
out = torch.argmax(out, dim = 1)
out = out[0]
viz = np.zeros([1024,1920,3],dtype = np.uint8)
for i in range(1024):
    for j in range(1920):
            if out[i,j] ==1: 
                viz[i,j,1]= 128
            if out[i,j] ==2:
                viz[i,j,0]=  127 
            if out[i,j] ==0:
                viz[i,j,:] = 0  

viz = Image.fromarray(viz)
viz = g_viz.save('predict.png')