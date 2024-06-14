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
import torch.nn as nn
import torch.mm as mm 
from torch import Tensor
from torch.nn import Parameter

class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size,):
        super().__init__()
        self.size = input_size
        self.hidden_size = hidden_size

        # Each gate consist of 4 Parameters a input weights, hidden weights, and bias input bias and hidden bias
        # input gate
        self.w_ii = Parameter(Tensor(input_size, hidden_size))
        self.w_hi = Parameter(Tensor(input_size, hidden_size))
        self.b_ii = Parameter(Tensor(hidden_size))
        self.b_hi = Parameter(Tensor(hidden_size))
        # forget gate
        self.w_if = Parameter(Tensor(input_size, hidden_size))
        self.w_hf = Parameter(Tensor(input_size, hidden_size))
        self.b_if = Parameter(Tensor(hidden_size))
        self.b_hf = Parameter(Tensor(hidden_size))
        # cell gate
        self.w_ig = Parameter(Tensor(input_size, hidden_size))
        self.w_hg = Parameter(Tensor(input_size, hidden_size))
        self.b_ig = Parameter(Tensor(hidden_size))
        self.b_hf = Parameter(Tensor(hidden_size))
        # output gate
        self.w_io = Parameter(Tensor(input_size, hidden_size))
        self.w_ho = Parameter(Tensor(input_size, hidden_size))
        self.b_ho = Parameter(Tensor(hidden_size))
        self.b_ho = Parameter(Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, input, hidden):

        # in this xt == input
                  # hx == ht-1
                  # cx == ct-1 

        hx, cx = hidden
        
        






'''
Symbol Definitions:
    c : cell state 
    n : normalizer state
    h : hidden state
    z : cell input
    i : input gate
    f : forget gate
    o : output gate
    \odot : Hadamard product
    \sigma : sigmoid
    \exp : exponent
    \tanh : tangent

'''