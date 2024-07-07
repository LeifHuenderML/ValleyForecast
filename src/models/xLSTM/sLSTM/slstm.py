"""
 * @file lstm.py
 * @author Leif Huender
 * @brief V1 is heavily inspired from https://github.com/georgeyiasemis/Recurrent-Neural-Networks-from-scratch-using-PyTorch
          V2 is more of a custom implementation with a focus on making it into the same lstm that we found to be most performan in gridsearch
 * @version 0.1
 * @date 2024-06-18
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
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable

class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True): 
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.xh = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.hh = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hidden=None):

        #initializes hidden for if the cell is the first layer
        if hidden is None:
            hidden = Variable(input.new_zeros(input.size(0), self.hidden_size))
            hidden = (hidden, hidden, hidden, hidden)

        h_t_p, c_t_p, m_t_p, n_t_p = hidden

        #computes the combined input to the input projection and the recurrent projection
        gates = self.xh(input) + self.hh(h_t_p)
        

        #slits the gates into 4 equal parts along the second dimension (1)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        #apply the activation functions to the 4 gates
        i_t = torch.exp(input_gate)
        f_t = torch.exp(forget_gate)
        z_t = torch.tanh(cell_gate) #cell input
        o_t = torch.sigmoid(output_gate)

        m_t = torch.max(torch.log(f_t) + m_t_p, torch.log(i_t))

        i_t = torch.exp(input_gate - m_t)
        f_t = torch.exp(torch.log(f_t) + m_t_p - m_t)

        c_t = f_t * c_t_p + i_t * z_t
        n_t = f_t * n_t_p + i_t
        h_t = o_t * (c_t / n_t)
        return h_t, (h_t, c_t, m_t, n_t)

    
class rigid_sLSTM(nn.Module):

    def __init__(self, input_size=19, hidden_size=200, num_layers=2, bias=True, output_size=1):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        self.l1 = sLSTMCell(input_size, hidden_size, bias)
        self.l2 = sLSTMCell(hidden_size, hidden_size, bias)

        # Create the Linear layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)

    def forward(self, input, hidden=None):
        batch_size = input.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().cuda()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().cuda()
        m0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().cuda()
        n0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().cuda()

        hidden = [(h0[i], c0[i], m0[i], n0[i]) for i in range(self.num_layers)]
        
        outputs = []
        for time_step in range(input.size(1)):  
            x = input[:, time_step, :].cuda()
            for i, lstm in enumerate([self.l1, self.l2]):
                x, hidden[i] = lstm(x, hidden[i])

            outputs.append(x.unsqueeze(1))

        out = torch.cat(outputs, dim=1)[:, -1, :]  

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = out.squeeze()  
        return out
    



    
class sLSTM(nn.Module):

    def __init__(self, input_size=19, hidden_size=200, bias=True):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.l = sLSTMCell(hidden_size, hidden_size, bias).to('cuda')

    def forward(self, input, hidden=None):
        batch_size = input.size(0)

        h0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()
        c0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()
        m0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()
        n0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()

        hidden = (h0, c0, m0, n0)        

        outputs = []
        for time_step in range(input.size(1)):  
            x = input[:, time_step, :].cuda()
            x, hidden = self.l(x, hidden)

            outputs.append(x.unsqueeze(1))

        # out = torch.cat(outputs, dim=1)[:, -1, :]  
        return outputs
    











