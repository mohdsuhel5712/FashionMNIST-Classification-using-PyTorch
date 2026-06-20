import torch
import torch.nn as nn

# making a class for model 
# actual traing model making of class
class fashionNet(nn.Module):
      # constructor
      def __init__(self):
            super().__init__()
            
            self.network = nn.Sequential(
                  nn.Flatten(),
                  nn.Linear(28*28,256),
                  nn.ReLU(),
                  
                  nn.Linear(256,128),
                  nn.ReLU(),
                  
                  nn.Linear(128,10)
            )
      def forward(self,x):
            return self.network(x)

