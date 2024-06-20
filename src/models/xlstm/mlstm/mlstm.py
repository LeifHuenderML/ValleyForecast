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
import numpy as np
import torch.nn as nn


class mLSTMcell(nn.Module):
    r'''
    mLSTMCell is a single cell from the xLSTM paper. This implementation does not utilize its parallelizability capabilities as it is beyond the scope of this research.

    Parameters:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden states features.
        bias (float): A constant added to the weights during the activation functions.
        name (str): The name of the cell, useful for identifying the internal structure when combined with other cells.

    State:
        C_t (Tensor): The cell state, a matrix used to encode information between cells at time t.
        n_t (Tensor): The normalizer state, used to regulate the information passed to the hidden state.
        h_t (Tensor): The hidden state, used to encode information similarly to the cell state.
        m_t (Tensor): The stabilizer state used to prevent overflow for the exponent activation functions.
        C_t_p (Tensor): The previous cell state.
        n_t_p (Tensor): The previous normalizer state.
        h_t_p (Tensor): The previous hidden state.
        m_t_p (Tensor): The previous stabilizer state.

    Inputs:
        q_t (Tensor): Query input, used for passing information between the input and the hidden state.
        k_t (Tensor): Key input, used for passing information between the input and the normalizer state.
        v_t (Tensor): Value input, used for passing information between the input and the cell state.
        x_t (Tensor): Input vector, the data passed to the mLSTM that it is trying to learn to approximate.

    Gates:
        i_t (Tensor): Input gate, controls what information gets passed between the input and the hidden states.
        f_t (Tensor): Forget gate, controls what information is forgotten between the input and hidden states.
        o_t (Tensor): Output gate, controls what information gets passed to the output from the input.
        stabil_i_t (Tensor): Stabilized input gate.
        stabil_f_t (Tensor): Stabilized forget gate.

    Dimensionality:
        d (int): The dimensionality of the key input.

    Weights:
        w_i (Tensor): Input gate weights.
        w_f (Tensor): Forget gate weights.
        w_o (Tensor): Output gate weights.
        w_q (Tensor): Query input weights.
        w_k (Tensor): Key input weights.
        w_v (Tensor): Value input weights.

    Biases:
        b_i (Tensor): Input gate bias.
        b_f (Tensor): Forget gate bias.
        b_o (Tensor): Output gate bias.
        b_q (Tensor): Query input bias.
        b_k (Tensor): Key input bias.
        b_v (Tensor): Value input bias.

    Symbols:
        \frac: Fraction.
        \sqrt: Square root.
        \exp: Exponent.
        \sigma: Sigmoid.
        \max: Maximum.
        \odot: Hadamard product.
    
    Forward Pass:
        \begin{align}
            C_t &= f_t C_{t-1} + i_t v_t k_t^T && \text{cell state (19)} \\
            n_t &= f_t n_{t-1} + i_t k_t && \text{normalizer state (20)} \\
            h_t &= o_t \odot \tilde{h}_t , \quad \tilde{h}_t = \frac{C_t q_t}{\max \left\{ |n_t^T q_t|, 1 \right\}} && \text{hidden state (21)} \\
            q_t &= W_q x_t + b_q && \text{query input (22)} \\
            k_t &= \frac{1}{\sqrt{d}} W_k x_t + b_k && \text{key input (23)} \\
            v_t &= W_v x_t + b_v && \text{value input (24)} \\
            i_t &= \exp(\tilde{i}_t) , \quad \tilde{i}_t = w_i^T x_t + b_i && \text{input gate (25)} \\
            f_t &= \sigma(\tilde{f}_t) \text{ OR } \exp(\tilde{f}_t) , \quad \tilde{f}_t = w_f^T x_t + b_f && \text{forget gate (26)} \\
            o_t &= \sigma(\tilde{o}_t) , \quad \tilde{o}_t = W_o x_t + b_o && \text{output gate (27)}
        \end{align}
    '''

    def __init__(self, input_size, hidden_size, bias=True, name="mLSTMCell"):
        super(mLSTMcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.name = name
        self.w = nn.Linear(input_size, hidden_size * 6, bias=True)

        self.reset_parameters()
        self.display_features()
        
    def forward(self,x_t, C_t_p, n_t_p, m_t_p ):

        gates = self.w(x_t)

        # line 22 and 24 are automatically computed here
        q_t, key_input, v_t, input_gate, forget_gate, output_gate = gates.chunk(6, 1)

        # line 23 from the paper
        d = self.hidden_size
        k_t = (1 / torch.sqrt(torch.tensor(d, dtype=torch.float32))) * key_input

        # line 25-27 from the paper
        i_t = torch.exp(input_gate)
        f_t = torch.exp(forget_gate)
        o_t = torch.sigmoid(output_gate)

        # line 15-17 from the paper, not included in the formula but it is stated in the paper that the stabilizer is applied in the mLSTM
        m_t = torch.max(torch.log(f_t) + m_t_p, torch.log(i_t))
        stabil_i_t = torch.exp(torch.log(i_t) - m_t)
        stabil_f_t = torch.exp(torch.log(f_t) + m_t_p - m_t)

        # line 19
        C_t = stabil_f_t * C_t_p + stabil_i_t * v_t * k_t
        # line 20
        n_t = stabil_f_t * n_t_p + stabil_i_t * k_t
        # line 21
        h_t_intermediate = (C_t * q_t) / torch.max(torch.abs(n_t.T @ q_t), torch.tensor(1.0, dtype=torch.float32))
        h_t = o_t * h_t_intermediate

        return h_t, C_t, n_t, m_t
    


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
