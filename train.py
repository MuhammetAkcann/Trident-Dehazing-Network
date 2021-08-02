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
from torchvision import transforms
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
import pandas as pd 


from fft import FFT
from brelu import BRELU


cuda0 = torch.cuda.set_device(1)
print("torch.cuda.current_device():",torch.cuda.current_device())

path = os.getcwd()
imagesPath = os.path.join(path,"patchedDataSet")

trainNames = sorted(os.listdir(imagesPath))

trainHazy = [i for i in trainNames if "hazy" in i]
trainGT = [i for i in trainNames if "GT" in i]


df = pd.DataFrame(columns = ["hazyPath","GTPath"])

for i in range(len(trainHazy)):
    df.loc[i] = [trainHazy[i]] + [trainGT[i]]

#print(df)

net = Net(pretrained=False)
checkpoint = torch.load("TDN_NTIRE2020_Dehazing.pt")

net.load_state_dict(checkpoint)
net.eval()
net = nn.DataParallel(net, device_ids=[1]).cuda()


criterion = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



class hazyGtDataset(Dataset):
    """
      The Class will act as the container for our dataset. It will take your dataframe, the root path, and also the transform function for transforming the dataset.
    """
    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        output_img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        
        image = Image.open(img_name)
        ouputImage = Image.open(output_img_name)
        
        if not self.transform:
            image = ToTensor()(image)
            image = Variable(image).cuda().unsqueeze(0)
            ouputImage = ToTensor()(ouputImage)
            ouputImage = Variable(ouputImage).cuda().unsqueeze(0)
        
        return (image.squeeze(), ouputImage.squeeze())

# INSTANTIATE THE OBJECT
train = hazyGtDataset(
    data_frame=df,
    root_dir=imagesPath
)

trainloader = torch.utils.data.DataLoader(train, batch_size=1,
                                          shuffle=True, num_workers=0)



for epoch in range(200):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        hazy, GT = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(hazy)
        #outputs, GT = outputs.squeeze(), GT.squeeze() 
        loss = criterion(outputs, GT)
        loss.backward()
        optimizer.step()
        print(str(i)+"th run, loss:",loss)
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

torch.save(net.state_dict(),"mseWeights.pt")
print('Finished Training')



"""
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter()

dataiter = iter(trainloader)
hazy, GT = dataiter.next()
im = net(hazy).squeeze(0)
im = im.cpu().data
im = ToPILImage()(im)
im.save("/home/muhammet.akcan/Trident-Dehazing-Network/x.png")
"""


#writer.add_graph(net, hazy)
#writer.close()

#writer.add_graph(net, train[0])
#writer.close()




"""
dataiter = iter(trainloader)
hazy, GT = dataiter.next()
hazy = hazy.cpu().data.squeeze()
print(hazy.shape)
hazy = ToPILImage()(hazy)
plt.imshow(hazy)
plt.title("hazy")
plt.axis('off')

plt.show()

GT = GT.cpu().data.squeeze()
GT = ToPILImage()(GT)
plt.imshow(GT)
plt.title("GT")
plt.axis('off')

plt.show()
"""