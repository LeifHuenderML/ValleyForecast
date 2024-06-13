"""
 * @file test.py
 * @author Leif Huender
 * @brief 
 * @version 0.1
 * @date 2024-06-13
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
import torch.nn as nn

class xLSTM(nn.Module):
    def __init__(self, input_size=19, hidden_size=256):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.m1 = mLstm(input_size, hidden_size)
        self.m2 = mLstm(hidden_size, hidden_size)
        
        self.s1 = sLSTM(hidden_size, hidden_size)
        self.s2 = sLSTM(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size/2)
        self.fc2 = nn.Linear(hidden_size/2, hidden_size/4)
        self.fc3 = nn.Linear(hidden_size/4, 1)

    def forward(self, x):
        h_0 = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(x.size(0), self.hidden_size).to(x.device)

        out, (h_n, c_n) = self.m1(x, (h_0, c_0))
        out = self.dropout(out)
        out, (h_n, c_n) = self.s1(out, (h_n, c_n))
        out = self.dropout(out)
        out, (h_n, c_n) = self.m2(out, (h_n, c_n))
        out = self.dropout(out)
        out, (h_n, c_n) = self.s2(out, (h_n, c_n))
        out = self.fc1(h_n) # pass the hidden state to the firs linear layer
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = out.flatten(start_dim=1)  
        return out
