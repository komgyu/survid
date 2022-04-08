
import numpy as np

g_viz = np.array([[1,2,0,0],[2,0,0,0]])

s_gt = (g_viz[:,:] != 0)*128
print (s_gt)