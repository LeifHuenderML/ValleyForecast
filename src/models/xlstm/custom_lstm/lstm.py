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


class LSTMCell(nn.Module):
    '''
    LSTMCell is a custom cell for an lstm created by Leif Huender with inspiration from https://github.com/georgeyiasemis/Recurrent-Neural-Networks-from-scratch-using-PyTorch

    This is the building block fro the lstm class

    Parameters:
    input_size: the number of features that the LSTMCell should expect
    hidden_size: the number of hidden features the LSTMCell should create
    bias: defaults to true, add the bias to the weighted sum of inputs before applying the activation function

    Variables:
    xh: input projection that takes the input_size and projects it to the hidden size * 4. one for every gate
    hh: recurrent projection that takes the hidden_size and projects it to the hidden size * 4. one for every gate
    '''
    def __init__(self, input_size, hidden_size, bias=True): 
        super(LSTMCell, self).__init__()
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
        r'''
        Inputs:
              input: of shape (batch_size, input_size)
              hidden: of shape (batch_size, hidden_size)
        Outputs:
              hy: of shape (batch_size, hidden_size)
              cy: of shape (batch_size, hidden_size)

        math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_y = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_y = o_t \odot \tanh(c_t) \\
        \end{array}
        '''

        #initializes hidden for if the cell is the first layer
        if hidden is None:
            hidden = Variable(input.new_zeros(input.size(0), self.hidden_size))
            hidden = (hidden, hidden)

        hx, cx = hidden

        #computes the combined input to the input projection and the recurrent projection
        gates = self.xh(input) + self.hh(hx)

        #slits the gates into 4 equal parts along the second dimension (1)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        #apply the activation functions to the 4 gates
        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        #compute the new cell state
        cy = cx * f_t + i_t * g_t
        #compute the new hidden state
        hy = o_t * torch.tanh(cy)

        return (hy, cy)

    
class LSTM(nn.Module):
    '''
    Uses the LSTMCell class and nn.Linear class to contruct a regressor LSTM

    Parameters:
    input_size: the number of features that the LSTM should expect
    hidden_size: the number of hidden features the LSTM should create
    bias: defaults to true, add the bias to the weighted sum of inputs before applying the activation function
    num_layers: number of layers the LSTM will have in the stack
    output_size: number of values to predict, since we are training it as a regressor we will 

    '''
    def __init__(self, input_size=19, hidden_size=265, num_layers=2, bias=True, output_size=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        #creating a list of LSTMCells to layer the LSTM stack
        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(LSTMCell(self.input_size,
                                            self.hidden_size,
                                            self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(LSTMCell(self.hidden_size,
                                                self.hidden_size,
                                                self.bias))

        #create the Linear layers
        self.fc1 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.fc2 = nn.Linear(int(self.hidden_size/2), int(self.hidden_size/4))
        self.fc3 = nn.Linear(int(self.hidden_size/4), self.output_size)

    def forward(self, input, hidden=None):
        '''
        Parameters: 
        input: the data 
        hidden: the hidden states

        Input of shape (batch_size, seqence length , input_size)
        Output of shape (batch_size, output_size)
        '''
        #initializes the models hidden states if there is none
        if hidden is None:
            h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
        else:
             h0 = hidden
        #stores the models outputs value
        outputs = []
        #initialize the hiden states for each layer
        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))
        #iterates over the time steps of the input sequence
        for time_step in range(input.size(1)):
            #iterates over the layers of the LSTM
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        input[:, time_step, :],
                        (hidden[layer][0],hidden[layer][1])
                        )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                        )

                hidden[layer] = hidden_l

            outputs.append(hidden_l[0])
        #reshape this to make it fit the linear layers
        out = outputs[-1].squeeze()
        #run through each of the linear layers
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
    
















class LSTMCellv2(nn.Module):
    '''
    LSTMCell is a custom cell for an lstm created by Leif Huender with inspiration from https://github.com/georgeyiasemis/Recurrent-Neural-Networks-from-scratch-using-PyTorch

    This is the building block fro the lstm class

    Parameters:
    input_size: the number of features that the LSTMCell should expect
    hidden_size: the number of hidden features the LSTMCell should create
    bias: defaults to true, add the bias to the weighted sum of inputs before applying the activation function

    Variables:
    xh: input projection that takes the input_size and projects it to the hidden size * 4. one for every gate
    hh: recurrent projection that takes the hidden_size and projects it to the hidden size * 4. one for every gate
    '''
    def __init__(self, input_size, hidden_size, bias=True): 
        super(LSTMCellv2, self).__init__()
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
        r'''
        Inputs:
              input: of shape (batch_size, input_size)
              hidden: of shape (batch_size, hidden_size)
        Outputs:
              hy: of shape (batch_size, hidden_size)
              cy: of shape (batch_size, hidden_size)

        math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_y = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_y = o_t \odot \tanh(c_t) \\
        \end{array}
        '''

        #initializes hidden for if the cell is the first layer
        if hidden is None:
            hidden = Variable(input.new_zeros(input.size(0), self.hidden_size))
            hidden = (hidden, hidden)

        hx, cx = hidden

        #computes the combined input to the input projection and the recurrent projection
        gates = self.xh(input) + self.hh(hx)

        #slits the gates into 4 equal parts along the second dimension (1)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        #apply the activation functions to the 4 gates
        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        #compute the new cell state
        cy = cx * f_t + i_t * g_t
        #compute the new hidden state
        hy = o_t * torch.tanh(cy)

        return hy, (hy, cy)

    
class LSTMv2(nn.Module):
    '''
    Uses the LSTMCell class and nn.Linear class to contruct a regressor LSTM

    Parameters:
    input_size: the number of features that the LSTM should expect
    hidden_size: the number of hidden features the LSTM should create
    bias: defaults to true, add the bias to the weighted sum of inputs before applying the activation function
    num_layers: number of layers the LSTM will have in the stack
    output_size: number of values to predict, since we are training it as a regressor we will 

    '''
    def __init__(self, input_size=19, hidden_size=200, num_layers=2, bias=True, output_size=1):
        super(LSTMv2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        self.l1 = LSTMCellv2(input_size, hidden_size, bias)
        self.l2 = LSTMCellv2(hidden_size, hidden_size, bias)

        # Create the Linear layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)

    def forward(self, input, hidden=None):
        batch_size = input.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().cuda()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().cuda()

        hidden = [(h0[i], c0[i]) for i in range(self.num_layers)]
        
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
    















