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
    sLSTMCell is a single cell from the xLSTM paper.

    Parameters:
        input_size (int): The number of features that the LSTMCell should expect.
        hidden_size (int): The number of hidden features the LSTMCell should create.
        bias (bool): Defaults to True. Adds the bias to the weighted sum of inputs before applying the activation function.

    States:
        c_t (Tensor): Cell state. The cell state acts as the memory component by storing and maintaining information over long sequences.
        n_t (Tensor): Normalizer state. The normalizer state modulates the influence of the forget and input gates over time.
        h_t (Tensor): Hidden state. The hidden state acts as the memory component by storing and maintaining information over long sequences.
        m_t (Tensor): Stabilizer state. Stabilizes the input and forget gates to prevent overflow caused by the exponential activation function inside them.

        C_t_p (Tensor): Previous cell state. Stores the cell state at the previous time step t-1.
        n_t_p (Tensor): Previous normalizer state. Stores the previous normalizer state at time step t-1.
        h_t_p (Tensor): Previous hidden state. Stores the previous hidden state at time step t-1.
        m_t_p (Tensor): Previous stabilizer state. Stores the previous stabilizer state at time step t-1.

    Inputs:
        x_t (Tensor): Input vector. Represents the input features at time step t.

    Gates:
        i_t (Tensor): Input gate. Controls what new information from the current input and previous hidden cell state will be added to the cell state.
        f_t (Tensor): Forget gate. Determines what information from the previous cell state should be forgotten or retained for the current time step.
        o_t (Tensor): Output gate. Determines what parts of the cell state should be used to compute the hidden state.
        stabil_i_t (Tensor): Stabilized input gate. The input gate with added stabilization.
        stabil_f_t (Tensor): Stabilized forget gate. The forget gate with added stabilization.

    Gating Factors:
        r_z (Tensor): Gating factor for cell input.
        r_i (Tensor): Gating factor for input gate.
        r_f (Tensor): Gating factor for forget gate.
        r_o (Tensor): Gating factor for output gate.

    Weights:
        w_i (Tensor): Input gate weights.
        w_f (Tensor): Forget gate weights.
        w_o (Tensor): Output gate weights.

    Biases:
        b_i (Tensor): Input gate bias.
        b_f (Tensor): Forget gate bias.
        b_o (Tensor): Output gate bias.

    Symbols:
        \sigma: Sigmoid function.
        \odot: Hadamard product.
        ^T: Matrix transpose.
        \exp: Exponent.
        \frac: Fraction.
        \sqrt: Square root.
        \log: Logarithm.
        \tanh: Tanh.
        \max: Maximum.

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

    def __init__(self, input_size, hidden_size, bias=True, name='sLSTMCell'): 
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.name = name
        self.w = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.r = nn.Linear(hidden_size, hidden_size * 4, bias=bias)

        self.reset_parameters()

        self.display_features()

    def forward(self, input, hidden=None, block_diags=None):
        if hidden is None:
            batch_size = input.size(0)
            hidden = (
                torch.zeros(batch_size, self.hidden_size, device=input.device),
                torch.zeros(batch_size, self.hidden_size, device=input.device),
                torch.zeros(batch_size, self.hidden_size, device=input.device),
                torch.zeros(batch_size, self.hidden_size, device=input.device)
            )
        c_t_p, h_t_p, n_t_p, m_t_p = hidden
            
        #encode the weights with the input with the hidden with the bias
        gates = self.w(input) + self.r(h_t_p)
        # split into their respective gates
        if block_diags == None:
            cell_input, input_gate, forget_gate, output_gate = gates.chunk(4,1)
        else:
            input_gate, forget_gate, cell_input, output_gate = block_diags

        #apply the activation functions
        z_t = torch.tanh(cell_input)
        i_t = torch.exp(input_gate)
        f_t = torch.exp(forget_gate)
        o_t = torch.sigmoid(output_gate)

        #creathe the stabilizer state
        m_t = torch.max((torch.log(f_t) + m_t_p), torch.log(i_t))

        #apply the stabilize fate  to the inpud and forget gates
        stabil_i_t = torch.exp(torch.log(i_t) - m_t)
        stabil_f_t = torch.exp(torch.log(f_t) + m_t_p - m_t)
        #update the states
        c_t = stabil_f_t * c_t_p + i_t * z_t
        n_t = stabil_f_t * n_t_p + stabil_i_t
        h_t = o_t * (c_t / n_t)

        #pass forward the new states
        return h_t, c_t, h_t, n_t, m_t

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
''''






'''
### TODO rewrite this it is not a true gated mlp
class GatedMLP(nn.Module):
    def __init__(self, input_size, output_size, projection_factor=4/3):
        super(GatedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, int(output_size * projection_factor))
        self.gate = nn.Linear(output_size, output_size)
    
    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        return x * gate
''''






'''
class sLSTMBlock(nn.Module):
    '''
    sLSTMBlock constructs a block for a stacked sLSTM model with normalization, convolution, sLSTM, and gated MLP layers.

    Parameters:
    - input_size (int): The expected input size for the block.
    - hidden_size (int): The hidden size for the sLSTM layer.
    - bias (bool): If True, introduces a bias to the sLSTM layer (default: True).
    - name (str): Used for identifying the model when built into a stack.

    Variables:
    - layer_norm (nn.LayerNorm): Normalizes the inputs to each layer to enhance generalization.
    - conv4 (nn.Conv1d): 1D convolutional layer with window size 4.
    - sLSTM_cell (sLSTMCell): The sLSTM layer.
    - group_norm (nn.GroupNorm): Divides the channels of the layer into several groups and normalizes the activation within each group separately.
    - gate_mlp_1-3 (GatedMLP): Uses the GatedMLP to capture spatial interactions across the sequence elements without using attention mechanisms.
    '''
    def __init__(self,input_size, hidden_size, bias=True, name="sLSTMBlock", ):
        super(sLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.name = name

        self.layer_norm = nn.LayerNorm(input_size)
        
        in_channels=19
        out_channels=64
        kernel_size=4
        stride=1
        padding=1
        self.conv4 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)


        self.sLSTM_cell = sLSTMCell(input_size, hidden_size, bias, name+' sLSTMCell_1')
        self.group_norm = nn.GroupNorm(num_groups=1,num_channels=1)
        self.gate_mlp_1 = GatedMLP(input_size=1, output_size=1)
        self.gate_mlp_2 = GatedMLP(input_size=1, output_size=1)
        self.gate_mlp_3 = GatedMLP(input_size=1, output_size=1, projection_factor=3/4)

    def forward(self,x):
        '''
        Forward pass follows the schematic provided in the xLSTM paper on page 29.

        Input and Residual Structure:
        The input to the block is embedded within a pre-LayerNorm (LN) residual structure. 
        This structure helps in stabilizing the learning process by normalizing the input data.

        Causal Convolution with Swish Activation:
        Optionally, the input is passed through a causal convolution layer with a window size of 4. 
        This convolution layer includes a Swish activation function. 
        The Swish activation helps in smoothing the output and improving the model's performance.
        The causal convolution is applied to the input and forget gates.

        sLSTM Components:
        The sLSTM block includes four main components: input gate (i), forget gate (f), cell update (z), and output gate (o).
        Each of these gates processes the input through a block-diagonal linear layer.

        Block-Diagonal Linear Layer:
        Each gate (i, f, z, o) has its input processed through a block-diagonal linear layer. 
        This layer has four diagonal blocks or "heads," 
        corresponding to the recurrent gate pre-activations from the last hidden state.
        The circular arrows indicate the recurrent connections within each head, 
        showing the recurrent nature of the sLSTM with four heads.

        GroupNorm Layer:
        After processing through the sLSTM gates, the resulting hidden state undergoes GroupNorm (GN). 
        This normalization is done head-wise, similar to LayerNorm but applied to each of the four heads individually.

        Gated MLP with GeLU Activation:
        The output from the GroupNorm layer is then passed through a Gated MLP (Multi-Layer Perceptron). 
        The MLP uses a GeLU (Gaussian Error Linear Unit) activation function. 
        The projection factor (PF) of 4/3 is applied to up- and down-project the output to match parameters. 
        This step adjusts the dimensions of the output appropriately.

        Combining Outputs and Residual Connection:
        The outputs from the Gated MLP are combined and passed through another projection layer with a projection factor (PF) of 3/4.
        The final output is combined with the residual connection (indicated by the "+" symbol), 
        ensuring that the input is added to the output, maintaining the residual nature of the block.

        LayerNorm:
        The final combined output is passed through a LayerNorm layer (LN), 
        which normalizes the combined output before it is passed on to the next stage or block.
        '''

        ## TODO rewrite this it is not correct
        out = self.layer_norm(x)
        out1 = out.copy()

        out = self.conv4(out)
        out = nn.SiLU(out)

        i, f = out.chunk(2,1)
        z, o = out1.chunk(2,1)

        i, f, z, o = torch.block_diag(i), torch.block_diag(f), torch.block_diag(z), torch.block_diag(o), 

        out, _, _, _, _ = self.sLSTM_cell(x, hidden=None, block_diags=(i, f, z, o))

        out = self.group_norm(out)

        mlp1_out = self.gate_mlp_1(out)
        mlp2_out = self.gate_mlp_2(out)

        mlp2_out = nn.GELU(mlp2_out)

        out = mlp1_out * mlp2_out
        out = self.gate_mlp_3(out)

        out = out + x

        return out
''''






'''
class sLSTM(nn.Module):
    '''
    Constructs a regression sLSTM model using sLSTMCell and nn.Linear.

    Parameters:
    - input_size (int): Number of input features for the sLSTM.
    - hidden_size (int): Number of hidden units in the sLSTM.
    - bias (bool): If True, adds a learnable bias to the sLSTM cells (default: True).
    - num_layers (int): Number of stacked sLSTM layers.
    - output_size (int): Number of output values to predict.

    Variables:
    - dropout (nn.Dropout): Dropout layer with a dropout probability of 0.1, applied between fully connected layers.
    - relu (nn.ReLU): ReLU activation function applied between fully connected layers to introduce non-linearity.
    - l1 (sLSTMCell): First sLSTM cell, processing input features and producing hidden state.
    - l2 (sLSTMCell): Second sLSTM cell, taking hidden state from the first cell and producing further hidden states.
    - fc1 (nn.Linear): Fully connected layer transforming hidden state to half its size.
    - fc2 (nn.Linear): Fully connected layer transforming to a quarter of the previous size.
    - fc3 (nn.Linear): Fully connected layer producing the final output.

    Forward Pass:
    - Initializes hidden states and cell states to zeros.
    - Processes input through sLSTM cells for each time step.
    - Applies fully connected layers with ReLU and dropout.
    - Returns the final output after processing through the fully connected layers.
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

        self.l1 = sLSTMCell(input_size, hidden_size, bias, 'Cell 1')
        self.l2 = sLSTMCell(hidden_size, hidden_size, bias, 'Cell 2')

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

            #possible bugs with how the hidden states are being handled
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
''''






'''