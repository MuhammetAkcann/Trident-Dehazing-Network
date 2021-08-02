import argparse
import torch
torch.cuda.empty_cache()
import time,os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from TDN import Net

path = os.getcwd()
imagesPath = os.path.join(path,"NH-HAZE")

trainNames = sorted(os.listdir(imagesPath))

trainHazy = [i for i in trainNames if "hazy" in i]
trainGT = [i for i in trainNames if "GT" in i]

for i in range(len(trainHazy)):
	im = Image.open(os.path.join(imagesPath,trainHazy[i]))
	gtIm = Image.open(os.path.join(imagesPath,trainGT[i]))
	width, height = im.size
	for j in range(16):
		x = j % 4
		y = j // 4
		# Setting the points for cropped image
		left = x * width / 4 
		top = y * height / 4
		right = left + width / 4
		bottom = top + height / 4
		 
		# Cropped image of above dimension
		# (It will not change original image)
		im1 = im.crop((left, top, right, bottom))
		outputIm = gtIm.crop((left, top, right, bottom))
		imName = str(i) + "_" + str(j) + "_" + trainHazy[i] 
		outName = str(i) + "_" + str(j) + "_" + trainGT[i] 
		patchPath = os.path.join(path,"patchedDataSet")
		im1.save(os.path.join(patchPath, imName))
		outputIm.save(os.path.join(patchPath, outName))
	print(i)
