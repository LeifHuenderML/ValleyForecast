"""
 * @file cell.py
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
    def __init__(self, input_size=19, hidden_size=265, num_layers=2, bias=True, output_size=1):
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
    
















class sLSTMCell(nn.Module):
    r'''
    LSTMCell is a custom cell for an lstm created by Leif Huender with inspiration from https://github.com/georgeyiasemis/Recurrent-Neural-Networks-from-scratch-using-PyTorch

    This is the building block fro the lstm class

    Parameters:
    input_size: the number of features that the LSTMCell should expect
    hidden_size: the number of hidden features the LSTMCell should create
    bias: defaults to true, add the bias to the weighted sum of inputs before applying the activation function

    Variables:
    States:
        c_t: cell state
            The cell state acts as the memory component by storing and maintaining information over long sequences
        n_t: normalizer state
            The normalizer state modelates the infuence the forget and input gate over time
        h_t: hidden state
            The hidden state acts as the memory component by storing and maintaining information over long sequences
        m_t: stabilizer state
            stabilizes the input and forget gates so that there is not an overflow caused from the exponential activation function inside them

        C_t_p: previous cell state
            Stores the cells state at the previous time step t-1.
        n_t_p: previous normalizer state
            Stores the previous normalizer state at time step t-1.
        h_t_p: previous hidden state
            Store the previous hidden state at time step t-1.
        m_t_p: previous stabilizer state
            Store the previous stabilizer state at time step t-1.
    Inputs:
        x_t: input vector
            Represents the input features at time step t.
    Gates: 
        i_t: input gate
            Controls what new information from the current input and previous hidden cell state will be added to the cell state.
        f_t: forget gate
            Determines what information from the previous cell stae should be forgotten or retained fo the current time step.
        o_t: output gate
            Determines what parts of the cell state should be used to compute the hidden state.
        stabil_i_t: stabil. input gate
            The input gate with the added stabilization.
        stabil_f_t: stabil forget gate
            The forget gate withe the added stabilization.
    Gating Factors: 
        r_z: gating factor for cell input
        r_i: gating factor for input gate
        r_f: gating factor for forget gate
        r_o: gating factor for output gate
    Weights:
        w_i: input gate weights
        w_f: forget gate weights
        w_o: output gate weights
    Biases:
        b_i: input gate bias
        b_f: forget gate bias
        b_o: output gate bias
    Symbols:
        \sigma: sigmoid function 
        \odot: hadamard product
        ^T: matric transpose
        \exp: exponent
        \frac: fraction
        \sqrt: square root
        \log: logarithm
        \tanh: tanh
        \max: max
    Forward Pass:
        \begin{align}
        c_t &= f_t \odot c_{tp} + i_t \odot z_t \tag{8} \\
        n_t &= f_t \odot n_{tp} + i_t \tag{9} \\
        h_t &= o_t \odot \hat{h}_t, \quad \hat{h}_t = \frac{c_t}{n_t} \tag{10} \\
        \tilde{z}_t &= w_z^T x_t + r_z h_{tp} + b_z \tag{11} \\
        \tilde{i}_t &= w_i^T x_t + r_i h_{tp} + b_i \tag{12} \\
        \tilde{f}_t &= w_f^T x_t + r_f h_{tp} + b_f \tag{13} \\
        \tilde{o}_t &= w_o^T x_t + r_o h_{tp} + b_o \tag{14} \\
        m_t &= \max(\log(f_t) + m_{tp}, \log(i_t)) \tag{15} \\
        \tilde{i}'_t &= \exp(\log(i_t) - m_t) \tag{16} \\
        \tilde{f}'_t &= \exp(\log(f_t) + m_{tp} - m_t) \tag{17}
        \end{align}
    '''


    def __init__(self, input_size, hidden_size, bias=True): 
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias



    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hidden=None):
        return

    
class sLSTM(nn.Module):
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
        super(sLSTM, self).__init__()
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