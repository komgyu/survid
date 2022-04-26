import numpy as np
import cv2
import glob

files = glob.glob('./original_data/original/*.png')
i = 0 
while i < len(files):
	img = cv2.imread(files[i], 1)
	img_scale = cv2.resize(img, (480, 256), interpolation=cv2.INTER_AREA)
	imgname = files[i].replace('./original_data/original\\','')
	print(imgname)
	cv2.imwrite('./original_data/scale/original/'+ str(imgname), img_scale)
	i += 1

