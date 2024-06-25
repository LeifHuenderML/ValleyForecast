import torch
import torch.nn as nn
from models.xLSTM.old_block import xLSTMBlock

class xLSTM(nn.Module):
    def __init__(self, input_size=19, hidden_size=256, num_layers=1, dropout=0.1):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.m1 = xLSTMBlock(input_size, hidden_size, num_layers, dropout, lstm_type='mlstm')
        self.s1 = xLSTMBlock(hidden_size, hidden_size, num_layers, dropout, lstm_type='slstm')
        self.fc1 = nn.Linear(input_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc3 = nn.Linear(hidden_size//4, 1)

    def forward(self, x):
        self.m1(x)
        print(f"Input shape: {x.shape}")
        out, hidden = self.m1(x)
        # print(f"After m1 shape: {out.shape}")
        out, _ = self.s1(out, hidden)
        # print(f"After s1 shape: {out.shape}")
        out = self.fc1(out)
        # print(f"After fc1 shape: {out.shape}")
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # print(f"After fc2 shape: {out.shape}")
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        # print(f"After fc3 shape: {out.shape}")
        out = self.relu(out)
        out = self.dropout(out)
        out = out.squeeze(-1)
        # print(f"Final output shape: {out.shape}")
        return out
