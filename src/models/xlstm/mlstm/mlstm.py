"""
 * @file mlstm.py
 * @author Leif Huender
 * @brief contains the cell for the mLSTM
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
import torch.nn as nn


class mLSTMcell(nn.Module):
    '''
    mLSTMcell is a single cell from the xLSTM paper. The key difference is that it is not written to utilize its parralelizability cababilities because that is beyond the scope of this research

    Parameters: 
        input_size: the size of the input features
        hidden_size: the size of the hidden states features
        bias: a constant addded to the weights during the activation functions
        name: the name of the cell usefull for identifying the internal structure when combined with other cells
    State:
        C_t: the cell state
            Is a matric instead of a scalar value used to encode information between cells state at time t.
        n_t: normalizer state
            Used to reglate the information passed to the hidden state.
        h_t: hidden state
            Used to encode infromation like the cell state.
        C_t_p: previous cells state
        n_t_p: previous normalizers state
        h_t_p: previous hidden state
    Inputs:
        q_t: query input
            Used for passing information between the input and the hidden state
        k_t: key input
            Used for passin information between the input and the normalizer state
        v_t: value input
            Used for passing information between the input and the cell state
        x_t: input vector
            The data passed to the mLSTM that it is trying to learn to approximate.
    Gates:
        i_t: input gate
            Controlst what information gets passed between the input and the hidden states
        f_t: forget gate
            Controls what informtation is forgotten when between the input and hidden states
        o_t: output gate
            Controls what informatin get pass to the output from the input
    
        

    '''
    def __init__(self, name, input_size, hidden_size, bias=True):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.name = name

