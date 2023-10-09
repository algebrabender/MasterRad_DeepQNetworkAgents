import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F

class DQNCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.convolution_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fully_connected = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.convolution_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
    
    def feature_size(self):
        return self.convolution_layers(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)