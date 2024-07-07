import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from mLSTM.mlstm import mLSTM
from sLSTM.slstm import sLSTM
from torch.utils.data import DataLoader

class xLSTM(nn.Module):
    def __init__(self, input_size=19, hidden_size=256, stack_config='ms',dropout=0.1):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stack_config = stack_config
        self.lstms = self.init_lstms()
        self.lstm_layers = len(self.lstms)
        self.dropout = dropout
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Create the Linear layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)

    def init_lstms(self,):
        lstms = []
        for layer in self.stack_config:
            if layer == 'm':
                if len(lstms) == 0:
                    input_size = self.input_size
                else:
                    input_size = self.hidden_size
                lstm_layer = mLSTM(input_size, self.hidden_size)
            elif layer == 's':
                if len(lstms) == 0:
                    input_size = self.input_size
                else:
                    input_size = self.hidden_size
                    
                lstm_layer = sLSTM(input_size, self.hidden_size)
            else:
                raise ValueError(f'{layer} is not a valid stack config character')

            lstms.append(lstm_layer)
        return lstms

    def forward(self, x):
        for lstm in self.lstms:
            x = lstm(x)
        
        out = torch.cat(x, dim=1)[:, -1, :]  

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
    def __init__(self,model, epochs, train_loader, val_loader,loss_path='xLSTM_losses.txt'):
        super(Trainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.train_losses = []
        self.val_losses = []
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_train_epoch = 0
        self.best_val_epoch = 0
        self.path = loss_path 

    def train(self,):
        for epoch in range(1, self.epochs+1):
            # Training
            self.model.train()
            num_batches = len(self.train_loader)
            total_train_loss = 0
            for x, y in self.train_loader:
                x, y = x.to('cuda'), y.to('cuda')
                output = self.model(x)
                loss = self.best_train_epochloss_fn(output, y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_train_loss += loss.item()
            
            avg_train_loss = np.sqrt(total_train_loss / num_batches)
            self.train_losses.append(avg_train_loss)
            
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                best_train_epoch = epoch

            # Validation
            self.model.eval()
            num_val_batches = len(self.val_loader)
            total_val_loss = 0
            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to('cuda'), y.to('cuda')
                    output = self.model(x)
                    loss = self.best_train_epochloss_fn(output, y)
                    total_val_loss += loss.item()
            
            avg_val_loss = np.sqrt(total_val_loss / num_val_batches)
            self.val_losses.append(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_epoch = epoch

            print(f'Epoch {epoch} Train RMSE Loss: {avg_train_loss:.4f}, Val RMSE Loss: {avg_val_loss:.4f}')

            # Save results to a .txt file
            with open(self.path, 'w') as f:
                f.write(f'Best Training RMSE Loss: {best_train_loss:.4f} at epoch {best_train_epoch}\n')
                f.write(f'Best Validation RMSE Loss: {best_val_loss:.4f} at epoch {best_val_epoch}\n')
                f.write('\nEpoch-wise losses:\n')
                for epoch, (train_loss, val_loss) in enumerate(zip(self.train_losses, self.val_losses), 1):
                    f.write(f'Epoch {epoch}: Train RMSE = {train_loss:.4f}, Val RMSE = {val_loss:.4f}\n')




def main():
    train = torch.load('/home/intellect/Documents/Research/Current/ValleyForecast/data/cleaned/train.pt')
    val = torch.load('/home/intellect/Documents/Research/Current/ValleyForecast/data/cleaned/val.pt')
    batch_size = 32
    train_loader = DataLoader(train, batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=True, drop_last=True)
    model = xLSTM(stack_config='ss')
    model.to('cuda')

    trainer = Trainer(model, 1000, train_loader, val_loader)
    start = time.time()
    trainer.train()
    end = time.time()

    print(f'\nTotal training time on 100 epochs: {end-start}')  


if __name__ == "__main__":
    main()