"""
 * @file xlstm.py
 * @author Leif Huender
 * @brief 
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
import time
import numpy as np
import torch.nn as nn
from mlstm import mlstm
from slstm import slstm


class xLSTM(nn.Module):
    '''
    This is a xLSTM regressor meant to process sequential data and produce a regressor value. 
    It consists of 4 alternately stacked mLSTM blocks and sLSTM blocks folowed by a 3 layer mlp;
    that introduces nonlinearalities through relu and increased robustness through dropout.

    Parameters:
    - input_size (int): The expected input size for the model.
    - hidden_size (int): The hidden size that will be used in all the sLSTM and mLSTM blocks.
    - bias (bool): If True, introduces a bias to each of the mLSTM and sLSTM blocks (default: True).
    - name (str): Used for identifying the model (default: 'xLSTM').

    Variables:
    - dropout (nn.Dropout): Dropout layer with a dropout probability of 0.1, applied between fully connected layers.
    - relu (nn.ReLU): ReLU activation function applied between fully connected layers to introduce non-linearity.
    - mBlock1 (mLSTMBlock): First mLSTMBlock, processes input feature and produces a hidden state
    - sBlock1 (sLSTMBlock): First sLSTMBlock, processes the hidden state from mBlock1 and produces a new hidden state
    - mBlock2 (mLSTMBlock): Second mLSTMBlock, processes the hidden state from sBlock1 and produces a new hiden state
    - sBlock2 (sLSTMBlock): Second sLSTMBlock, processes the hidden state from mBlock1 and produces a new hiden state
    - fc1 (nn.Linear): Fully connected layer transforming hidden state to half its size.
    - fc2 (nn.Linear): Fully connected layer transforming to a quarter of the previous size.
    - fc3 (nn.Linear): Fully connected layer producing the final output.
    '''
    def __init__(self,input_size, hidden_size, bias=True, name='xLSTM'):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.name = name

        self.mBlock1 = mlstm.mLSTMBlock()
        self.mbBlock2 = mlstm.mLSTMBlock()
        
        self.sBlock1 = slstm.sLSTMBlock()
        self.sBlock2 = slstm.sLSTMBlock()

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.mBlock1(x)
        out = self.sBlock1(out)
        out = self.mbBlock2(out)
        out = self.sBlock2(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = out.squeeze()

        return out
    
class Trainer():
    def __init__(self, model, loader, epochs=100, checkpoint_path='best_model.pt', model_checkpoint=True, early_stopping=True, patience=10, delta=10):
        super(Trainer, self).__init__()
        self.model = model
        self.loader = loader
        self.epochs = epochs
        self.model_checkpoint = model_checkpoint
        self.early_stopping = early_stopping
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.delta = delta
        self.patience_tracker = 0
        self.losses = []
 
    def train(self):
        epoch = 1
        stop = False
        while epoch <= self.epochs and stop:
            num_batches = len(self.loader)
            total_loss = 0
            self.model.train()
            for x, y in self.loader:
                x, y = x.to('cuda'), y.to('cuda')
                output = self.model(x)
                loss = self.loss_fn(output,y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()

            
            avg_loss = total_loss/num_batches
            avg_loss = np.sqrt(avg_loss)

            self.checkpoint_model(avg_loss)

            stop = self.early_stop()

            self.losses.append(avg_loss)
            print(f'Epoch {epoch} RMSE Loss: {avg_loss}')
            epoch += 1
        print(f'Final Epoch {epoch} RMSE Loss: {avg_loss}')

    
    def checkpoint_model(self, loss):
        if loss < min(self.losses) and self.model_checkpoint:
                torch.save(self.model.state_dict(), self.checkpoint_path)

    def early_stop(self, loss):
        if self.early_stopping:
            if (loss + self.delta) > min(self.losses):
                self.patience_tracker += 1
            if self.patience_tracker == self.patience:
                return True
        return False
    

class Lego(nn.Module):
    '''
    Takes a string and parses it to create a xLSTM model with the defined block stack. 
    For instance smssm would create sLSTMBlock -> mLSTMBlock -> sLSTMBlock -> sLSTMBlock -> mLSTMBlock ->
    '''
    def __init__(self, input_size, hidden_size, bias, name):
        super(Lego, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.name = name

        self.block_layers = []

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def build(self, blocks):
        for index, block in enumerate(blocks):
            if block == 's':
                self.block_layers.append(slstm.sLSTMBlock(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    bias=self.bias,
                    name='sLSTMBlock_layer'+ str(index)
                ))
            elif block == 'm':
                self.block_layers.append(mlstm.mLSTMBlock(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    bias=self.bias,
                    name='mLSTMBlock_layer'+ str(index)
                ))
            else:
                raise ValueError(f"Invalid block value: {block}, block value can only be 'm' or 's'.")
        self.block_layers[0].input_size = self.input_size

    def forward(self, x):
        for layer in self.block_layers:
            x = layer(x)
            
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = out.squeeze()

        return out
        