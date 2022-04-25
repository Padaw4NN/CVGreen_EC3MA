"""
@author: Alexander Cardoso
"""

"""
        UNUSABLE... NOW


import torch
import torch.nn as nn

torch.manual_seed(4231)

class ClassifierModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv_1 = nn.Conv2d(in_channels=3 , out_channels=32, kernel_size=(3, 3))
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        
        self.fc1 = nn.Linear(in_features=512, out_features=512)
        self.out = nn.Linear(in_features=512, out_features=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.bnorm = nn.BatchNorm2d(num_features=32)
        self.dropout = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        
        x = self.max_pool(self.bnorm(self.relu(self.conv_1(x))))
        x = self.max_pool(self.bnorm(self.relu(self.conv_2(x))))
        
        x = nn.Flatten(x)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.out(x))
        x = self.sigmoid(x)
        
        return x

"""