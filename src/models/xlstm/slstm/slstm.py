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


    def __init__(self, name, input_size, hidden_size, bias=True): 
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.name = name
        self.w = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.r = nn.Linear(hidden_size, hidden_size * 4, bias=bias)

        self.reset_parameters()

        self.display_features()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def display_features(self,):
        print('='*100)
        print(f'{self.name} input and output features')
        print(f'Number of input features for w: {self.w.in_features}')
        print(f'Number of output features for w: {self.w.out_features}')
        print()
        print(f'Number of input features for r: {self.r.in_features}')
        print(f'Number of output features for r: {self.r.out_features}')
        print()


    def forward(self, input, c_t_p, h_t_p, n_t_p, m_t_p):
        #encode the weights with the input with the hidden with the bias
        gates = self.w(input) + self.r(h_t_p)
        print(f'gates shape {gates.shape}')
        # split into their respective gates
        cell_input, input_gate, forget_gate, output_gate = gates.chunk(4,1)


        #apply the activation functions
        z_t = torch.tanh(cell_input)
        i_t = torch.exp(input_gate)
        f_t = torch.exp(forget_gate)
        o_t = torch.sigmoid(output_gate)
        
        print(f'cell_input shape: {cell_input.shape}')
        print(f'input_gate shape: {input_gate.shape}')
        print(f'forget_gate shape: {forget_gate.shape}')
        print(f'output_gate shape: {output_gate.shape}')
        print()
        print(f'z_t shape: {z_t.shape}')
        print(f'i_t shape: {i_t.shape}')
        print(f'f_t shape: {f_t.shape}')
        print(f'o_t shape: {o_t.shape}')
        print()
        print(f'input shape: {input.shape}')
        print(f'c_t_p shape: {c_t_p.shape}')
        print(f'h_t_p shape: {h_t_p.shape}')
        print(f'n_t_p shape: {n_t_p.shape}')
        print(f'm_t_p shape: {m_t_p.shape}')


        #creathe the stabilizer state
        m_t = torch.max((torch.log(f_t) + m_t_p), torch.log(i_t))
        print()
        print(f'm_t shape: {m_t.shape}')

        #apply the stabilize fate  to the inpud and forget gates
        stabil_i_t = torch.exp(torch.log(i_t) - m_t)
        stabil_f_t = torch.exp(torch.log(f_t) + m_t_p - m_t)
        print()
        print(f'stabil_i_t shape {stabil_i_t.shape}')
        print(f'stabil_f_t shape {stabil_i_t.shape}')
        print()
        #update the states
        c_t = stabil_f_t * c_t_p + i_t * z_t
        n_t = stabil_f_t * n_t_p + stabil_i_t
        h_t = o_t * (c_t / n_t)

        print(f'c_t shape: {c_t.shape}')
        print(f'n_t shape: {n_t.shape}')
        print(f'h_t shape: {h_t.shape}')
        print()
        #pass forward the new states
        return h_t, c_t, h_t, n_t, m_t

    
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
    def __init__(self, input_size=19, hidden_size=200, num_layers=2, bias=True, output_size=1):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        self.l1 = sLSTMCell('Cell 1', input_size, hidden_size, bias)
        self.l2 = sLSTMCell('Cell 2', hidden_size, hidden_size, bias)

        # Create the Linear layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)

    def forward(self, input, hidden=None):
        batch_size = input.size(0)
        #hidden states
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().cuda()
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().cuda()
        n = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().cuda()
        m = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().cuda()
        
        outputs = []
        for time_step in range(input.size(1)):  
            x = input[:, time_step, :].cuda()

            #possible bugs with hor the hidden states are being handled
            out, c[0], h[0], n[0], m[0] = self.l1(x, c[0], h[0], n[0], m[0])
            out, c[1], h[1], n[1], m[1] = self.l1(out, c[1], h[1], n[1], m[1])
            
            outputs.append(out.unsqueeze(1))


        out = torch.cat(outputs, dim=1)[:, -1, :]  
        # print(f'out shape {out.shape}')
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = out.squeeze()  
        return out
    
