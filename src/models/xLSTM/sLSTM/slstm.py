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
import time
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable
from torch.utils.data import DataLoader

class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True): 
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.xh = nn.Linear(input_size, hidden_size * 4, bias=bias).cuda()
        self.hh = nn.Linear(hidden_size, hidden_size * 4, bias=bias).cuda()
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hidden):
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

    
class sLSTM(nn.Module):

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
    



    
# class sLSTM(nn.Module):

#     def __init__(self, input_size=19, hidden_size=200, bias=True):
#         super(sLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.bias = bias
        
#         self.l = sLSTMCell(hidden_size, hidden_size, bias).to('cuda')

#     def forward(self, input, hidden=None):
#         batch_size = input.size(0)

#         h0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()
#         c0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()
#         m0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()
#         n0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()

#         hidden = (h0, c0, m0, n0)        

#         outputs = []
#         for time_step in range(input.size(1)):  
#             x = input[:, time_step, :].cuda()
#             x, hidden = self.l(x, hidden)

#             outputs.append(x.unsqueeze(1))

#         # out = torch.cat(outputs, dim=1)[:, -1, :]  
#         return outputs
    









def main():
    def trainer(model, epochs, train_loader, val_loader, loss_fn, optim):
        train_losses = []
        val_losses = []
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        best_train_epoch = 0
        best_val_epoch = 0

        for epoch in range(1, epochs+1):
            # Training
            model.train()
            num_batches = len(train_loader)
            total_train_loss = 0
            for x, y in train_loader:
                x, y = x.to('cuda'), y.to('cuda')
                output = model(x)
                loss = loss_fn(output, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_train_loss += loss.item()
            
            avg_train_loss = np.sqrt(total_train_loss / num_batches)
            train_losses.append(avg_train_loss)
            
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                best_train_epoch = epoch

            # Validation
            model.eval()
            num_val_batches = len(val_loader)
            total_val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to('cuda'), y.to('cuda')
                    output = model(x)
                    loss = loss_fn(output, y)
                    total_val_loss += loss.item()
            
            avg_val_loss = np.sqrt(total_val_loss / num_val_batches)
            val_losses.append(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_epoch = epoch

            print(f'Epoch {epoch} Train RMSE Loss: {avg_train_loss:.4f}, Val RMSE Loss: {avg_val_loss:.4f}')

            # Save results to a .txt file
            with open('test_1_round_3_training_results.txt', 'w') as f:
                f.write(f'Best Training RMSE Loss: {best_train_loss:.4f} at epoch {best_train_epoch}\n')
                f.write(f'Best Validation RMSE Loss: {best_val_loss:.4f} at epoch {best_val_epoch}\n')
                f.write('\nEpoch-wise losses:\n')
                for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
                    f.write(f'Epoch {epoch}: Train RMSE = {train_loss:.4f}, Val RMSE = {val_loss:.4f}\n')

        return train_losses, val_losses, best_train_loss, best_val_loss, best_train_epoch, best_val_epoch


    train = torch.load('/home/intellect/Documents/Research/Current/ValleyForecast/data/cleaned/train.pt')
    val = torch.load('/home/intellect/Documents/Research/Current/ValleyForecast/data/cleaned/val.pt')
    batch_size = 32
    train_loader = DataLoader(train, batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=True, drop_last=True)
    model = sLSTM()
    model.to('cuda')
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=0.001)

    start = time.time()
    losses = trainer(model, 1000, train_loader, val_loader, loss_fn, optim)
    end = time.time()

    print(f'\nTotal training time on 100 epochs: {end-start}')



if __name__ == "__main__":
    main()
