## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.conv1 = nn.Conv2d(1, 8, 3)      
        self.conv2 = nn.Conv2d(8, 14, 3)
        self.conv3 = nn.Conv2d(14, 22, 3)
        self.conv4 = nn.Conv2d(22, 32, 3)

        self.dropout2d = nn.Dropout(p=0.5) 
        
        
        self.linear1 = nn.Linear(32*12*12,256)
        self.linear2 = nn.Linear(256 ,256)
        self.linear3 = nn.Linear(256,136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        #print("XXXXX", x.size())   #check the input size of fully connected layer
       
        x = x.view(-1, self.num_flat_features(x)) # batch size is excluded...
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout2d(x)
        
        x = self.linear2(x)      
        x = F.relu(x)
        x = self.dropout2d(x)
        
        x = self.linear3(x)   
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
