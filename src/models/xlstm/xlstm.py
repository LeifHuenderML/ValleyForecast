"""
 * @file xlstm.py
 * @author Leif Huender
 * @brief 
 * @version 0.1
 * @date 2024-06-20
 * 
 * @copyright Copyright (c) 2024 Leif Huender
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
"""

import torch
import time
import torch.nn as nn
from mlstm import mlstm
from slstm import slstm


class xLSTM(nn.Module):
    def __init__(self,input_size, hidden_size, bias=True, name='xLSTM'):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.name = name

        self.mBlock1 = mlstm.mLSTMBlock()
        self.mbBlock2 = mlstm.mLSTMBlock()
        
        self.sBlock1 = slstm.sLSTMBlock()
        self.sBlock2 = slstm.sLSTMBlock()

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.mBlock1(x)
        out = self.sBlock1(out)
        out = self.mbBlock2(out)
        out = self.sBlock2(out)
        out = self.fc1()


