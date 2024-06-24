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

    def forward(self, x_t, C_t_p, n_t_p, m_t_p ):

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
''''






'''
class mLSTMBlock(nn.Module):
    '''
    mLSTMBlock constructs a block for a stacked mLSTM model with normalization, convolution, mLSTM, and gated MLP layers.

    Parameters:
    - input_size (int): The expected input size for the block.
    - hidden_size (int): The hidden size for the mLSTM layer.
    - bias (bool): If True, introduces a bias to the mLSTM layer (default: True).
    - name (str): Used for identifying the model when built into a stack.

    Variables:
    - layer_norm (nn.LayerNorm): Normalizes the inputs to each layer to enhance generalization.
    - conv4 (nn.Conv1d): 1D convolutional layer with window size 4.
    - mLSTM_cell (mLSTMCell): The mLSTM layer.
    - group_norm (nn.GroupNorm): Divides the channels of the layer into several groups and normalizes the activation within each group separately.
    - gate_mlp_1-3 (GatedMLP): Uses the GatedMLP to capture spatial interactions across the sequence elements without using attention mechanisms.
    '''
    def __init__(self, input_size, hidden_size, bias=True, name='mLSTMBlock'):
        super(mLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.name = name

        self.layer_norm = nn.LayerNorm()
        self.conv4 = nn.Conv1d()
        self.mLSTM = mLSTMcell()
        self.group_norm = nn.GroupNorm()
        self.gated_mlp1 = GatedMLP()
        self.gated_mlp2 = GatedMLP()
        self.gated_mlp3 = GatedMLP(projection_factor=1/2)

    def forward(self, x):
        r'''
        Forward pass follows the schematic of the mLSTM block on page 30 of the xLSTM paper.

        Residual Structure and Input Processing
        Residual Structure: The mLSTM block is embedded in a pre-LayerNorm (LN) residual structure, 
        which helps stabilize the training by normalizing the inputs.
        Input Up-Projection: The input is first up-projected with a projection factor (PF) of 2. 
        This means that the dimensionality of the input is increased by a factor of 2.

        Projection for Output Gate and mLSTM Cells
        External Output Gate: The up-projected input is split into two parts. 
        One part is used for an externalized output gate, 
        and the other part is used as input for the mLSTM cells.

        Convolution and Skip Connection
        Causal Convolution: The input intended for the mLSTM cells undergoes a dimension-wise causal convolution 
        with a kernel size of 4 (Conv4). 
        This step ensures that the model maintains causal relationships within the sequence.
        Swish Activation: The convolved input is then activated using the Swish function,
        which helps in introducing non-linearity.
        Learnable Skip Connection (LSkip): There is a learnable skip connection which allows the input to bypass 
        the mLSTM cells and be directly added to the output after the mLSTM processing.

        Block-Diagonal Projection
        Input q and k: The inputs q (queries) and k (keys) are obtained via block-diagonal projection matrices of block size 4. 
        This means that the projection is applied separately to blocks of size 4 within the input.
        Values v: The values v are fed directly to the mLSTM cells, skipping the convolution part.

        mLSTM Sequence Mixing
        mLSTM Cells: The mLSTM cells, which are multi-headed (NH=4), perform the sequence mixing. 
        They process the q, k, and v inputs to produce the mixed sequence output.
        GroupNorm (GN): The outputs from the mLSTM cells are normalized using GroupNorm. 
        This is done in a head-wise manner, applying layer normalization separately to each of the four heads.

        Final Output Processing
        Adding Skip Connection: The learnable skip input is added to the normalized mLSTM output.
        Gated Component-Wise: The result is then gated component-wise with the external output gate. 
        This gating mechanism controls the flow of information based on the output gate's parameters.
        Down-Projection: Finally, the output is down-projected to its original dimensionality, 
        completing the processing within the mLSTM block.

        Summary
        The mLSTM block processes inputs through a combination of up-projection, causal convolution, and multi-headed sequence mixing. 
        It employs advanced normalization techniques like GroupNorm and a residual structure to stabilize training. 
        The use of learnable skip connections and gating mechanisms further enhances 
        its ability to capture and propagate important features through the network.
        '''
        
        out = self.layer_norm(x)
        gated_out1 = self.gated_mlp1(out)
        gated_out2 = self.gated_mlp2(out)

        conv_out = self.conv4(gated_out1)
        conv_out = nn.SiLU(conv_out)

        q, k = conv_out.chunk(2,1)
        
        out, _, _, _ = self.mLSTM(x, ''' fill in the blanks''', q, k, gated_out1)
        out = self.group_norm(out)
        out = out + gated_out1

        gated_out2 = nn.SiLU(gated_out2)

        out = out * gated_out2

        out = self.gated_mlp3(out)
        out = out + x
        return out
''''






'''
class GatedMLP(nn.Module):
    def __init__(self, input_size, output_size, projection_factor=2):
        super(GatedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, int(output_size * projection_factor))
        self.gate = nn.Linear(output_size, output_size)
    
    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        return x * gate