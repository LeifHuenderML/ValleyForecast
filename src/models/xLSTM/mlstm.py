import torch
import torch.nn as nn
import torch.nn.functional as F

class mLSTMCell(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_weights = nn.Linear(input_size, hidden_size*4, device='cuda')
        
