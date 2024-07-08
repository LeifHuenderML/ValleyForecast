import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from mLSTM.mlstm import mLSTMCell
from sLSTM.slstm import sLSTMCell
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

        self.s1 = sLSTMCell(input_size, hidden_size)
        self.m1 = mLSTMCell(hidden_size, hidden_size)

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
                lstm_layer = mLSTMCell(input_size, self.hidden_size)
            elif layer == 's':
                if len(lstms) == 0:
                    input_size = self.input_size
                else:
                    input_size = self.hidden_size
                    
                lstm_layer = sLSTMCell(input_size, self.hidden_size)
            else:
                raise ValueError(f'{layer} is not a valid stack config character')

            lstms.append(lstm_layer)
        return lstms
    
    def forward(self, input):
        batch_size = input.size(0)

        mh0 = torch.zeros(batch_size, self.hidden_size).requires_grad_().cuda()
        mc0 = torch.zeros(batch_size, self.hidden_size, self.hidden_size).requires_grad_().cuda()
        mm0 = torch.zeros(batch_size, self.hidden_size).requires_grad_().cuda()
        mn0 = torch.zeros(batch_size, self.hidden_size).requires_grad_().cuda()

        m_hidden = (mh0, mc0, mm0, mn0)   

        sh0 = torch.zeros(batch_size, self.hidden_size).requires_grad_().cuda()
        sc0 = torch.zeros(batch_size, self.hidden_size).requires_grad_().cuda()
        sm0 = torch.zeros(batch_size, self.hidden_size).requires_grad_().cuda()
        sn0 = torch.zeros(batch_size, self.hidden_size).requires_grad_().cuda()

        s_hidden = (sh0, sc0, sm0, sn0)        
        outputs = []
        # for time_step in range(input.size(1)):  
        #     x = input[:, time_step, :].cuda()
        #     x, s_hidden = self.s1(x, s_hidden)
        #     x, m_hidden = self.m1(x, m_hidden)
        for time_step in range(input.size(1)):  
            x = input[:, time_step, :].cuda()
            for i, lstm in enumerate(self.lstms):
                if self.stack_config[i] == 'm':
                    x, m_hidden = lstm(x, m_hidden)
                elif self.stack_config[i] == 's':
                    x, s_hidden = lstm(x, s_hidden)
                else: 
                    raise ValueError("improper comparison")

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

    # def forward(self, input):
    #     batch_size = input.size(0)

    #     mh0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()
    #     mc0 = torch.zeros(1, batch_size, self.hidden_size, self.hidden_size).requires_grad_().cuda()
    #     mm0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()
    #     mn0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()

    #     m_hidden = (mh0, mc0, mm0, mn0)   

    #     sh0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()
    #     sc0 = torch.zeros(1, batch_size, self.hidden_size, self.hidden_size).requires_grad_().cuda()
    #     sm0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()
    #     sn0 = torch.zeros(1, batch_size, self.hidden_size).requires_grad_().cuda()

    #     s_hidden = (sh0, sc0, sm0, sn0)        
    #     outputs = []
    #     for time_step in range(input.size(1)):  
    #         x = input[:, time_step, :].cuda()
    #         for i, lstm in enumerate(self.lstms):
    #             if self.stack_config[i] == 'm':
    #                 x, m_hidden = lstm(x, m_hidden)
    #             elif self.stack_config[i] == 's':
    #                 x, s_hidden = lstm(x, s_hidden)
    #             else: 
    #                 raise ValueError("improper comparison")



    #         outputs.append(x.unsqueeze(1))

    #     out = torch.cat(outputs, dim=1)[:, -1, :]  

    #     out = self.fc1(out)
    #     out = self.relu(out)
    #     out = self.dropout(out)
    #     out = self.fc2(out)
    #     out = self.relu(out)
    #     out = self.dropout(out)
    #     out = self.fc3(out)
    #     out = out.squeeze()  
    #     return out

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
                loss = self.loss_fn(output, y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_train_loss += loss.item()
            
            avg_train_loss = np.sqrt(total_train_loss / num_batches)
            self.train_losses.append(avg_train_loss)
            
            if avg_train_loss < self.best_train_loss:
                self.best_train_loss = avg_train_loss
                self.best_train_epoch = epoch

            # Validation
            self.model.eval()
            num_val_batches = len(self.val_loader)
            total_val_loss = 0
            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to('cuda'), y.to('cuda')
                    output = self.model(x)
                    loss = self.loss_fn(output, y)
                    total_val_loss += loss.item()
            
            avg_val_loss = np.sqrt(total_val_loss / num_val_batches)
            self.val_losses.append(avg_val_loss)
            
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_val_epoch = epoch

            print(f'Epoch {epoch} Train RMSE Loss: {avg_train_loss:.4f}, Val RMSE Loss: {avg_val_loss:.4f}')

            # Save results to a .txt file
            with open(self.path, 'w') as f:
                f.write(f'Best Training RMSE Loss: {self.best_train_loss:.4f} at epoch {self.best_train_epoch}\n')
                f.write(f'Best Validation RMSE Loss: {self.best_val_loss:.4f} at epoch {self.best_val_epoch}\n')
                f.write('\nEpoch-wise losses:\n')
                for epoch, (train_loss, val_loss) in enumerate(zip(self.train_losses, self.val_losses), 1):
                    f.write(f'Epoch {epoch}: Train RMSE = {train_loss:.4f}, Val RMSE = {val_loss:.4f}\n')




'''
The GridSearch class is designed for hyperparameter tuning of machine learning models, particularly those built using PyTorch. 
It systematically explores a range of hyperparameter combinations to determine the set that yields the best performance as measured by loss metrics. 
This class is particularly useful in optimizing model architectures and training routines.

Attributes:
    train_loader (DataLoader): DataLoader for the training dataset.
    val_loader (DataLoader): DataLoader for the validation dataset.
    train_loss_df (DataFrame): DataFrame to store training losses for each hyperparameter combination.
    val_loss_df (DataFrame): DataFrame to store validation losses for each hyperparameter combination.

Methods:
    search(param_dict):
        Executes a grid search over the specified hyperparameter space defined in param_dict. It evaluates each combination of parameters,
        updates the DataFrame with the loss for each configuration, and identifies the best parameter combination based on training and validation losses.

        Parameters:
            param_dict (dict): Dictionary with parameter names as keys and lists of parameter options as values.

        Returns:
            tuple: Contains minimum training loss, minimum validation loss, and the best hyperparameter combination.

    evaluate_model(**params):
        Evaluates a model with specified parameters. Sets up a model, defines its loss function and optimizer, and trains it using the Trainer class.
        This method is called internally by the search method to test each parameter combination.

        Parameters:
            params (dict): Dictionary of parameters required to build and train the model.

        Returns:
            tuple: Contains lists of training and validation losses for the evaluated model.

Usage Example:
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    param_dict = {
        'hidden_size': [128, 256],
        'num_layers': [1, 2],
        'bias': [True, False],
        'batch_first': [True],
        'dropout': [0.0, 0.2],
        'bidirectional': [True, False],
        'proj_size': [0, 64]
    }
    grid_search = GridSearch(train_loader, val_loader)
    best_train_loss, best_val_loss, best_params = grid_search.search(param_dict)
    print(f'Best Training Loss: {best_train_loss}')
    print(f'Best Validation Loss: {best_val_loss}')
    print(f'Best Parameters: {best_params}')
'''
class GridSearch():
    def __init__(self, train_loader, val_loader,):
        self.train_loader = train_loader
        self.val_loader =val_loader
        self.train_loss_df = pd.DataFrame()
        self.val_loss_df = pd.DataFrame()


    def search(self, param_dict):
        keys = param_dict.keys()
        values = param_dict.values()
        print(f'Search Space: {len(list(product(*values)))}')
        current_train_min = float('inf')
        current_val_min = float('inf')
        best_param_combo = ''
        # creates the cartesian product of all the combinations for searching over then loopss through them
        for combo in product(*values):
            param_combo = dict(zip(keys, combo))
            train_losses, val_losses = self.evaluate_model(**param_combo)
            self.train_loss_df = pd.concat([self.train_loss_df, pd.Series(train_losses)])
            self.val_loss_df = pd.concat([self.val_loss_df, pd.Series(val_losses)])
            if (min(train_losses) < current_train_min) and (min(val_losses) < current_val_min): 
                current_train_min = min(train_losses)
                current_val_min = min(val_losses)
                best_param_combo = param_combo
                file = open('search.txt', 'w')
                file.write('Min Train RMSE: ' + str(current_train_min) + '\nMin Validation RMSE: ' + str(current_val_min) + '\n' + str(best_param_combo))
                file.close()
        
        return current_train_min, current_val_min, best_param_combo

    def evaluate_model(self, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size):
        print( hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size)
        model = LSTM(input_size=19, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, proj_size=proj_size)
        loss_fn = nn.MSELoss()
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,)

        trainer = Trainer(model, self.train_loader, self.val_loader, loss_fn, optimizer=optimizer)
        _, train_losses, val_losses = trainer.train(epochs=100)
        return train_losses, val_losses
    




# def main():
#     train = torch.load('/home/intellect/Documents/Research/Current/ValleyForecast/data/cleaned/train.pt')
#     val = torch.load('/home/intellect/Documents/Research/Current/ValleyForecast/data/cleaned/val.pt')
#     batch_size = 32
#     train_loader = DataLoader(train, batch_size, shuffle=True, drop_last=True)
#     val_loader = DataLoader(val, batch_size=32, shuffle=True, drop_last=True)
#     model = xLSTM(stack_config='ss')
#     model.to('cuda')

#     trainer = Trainer(model, 1000, train_loader, val_loader)
#     start = time.time()
#     trainer.train()
#     end = time.time()

#     print(f'\nTotal training time on 100 epochs: {end-start}')  


# if __name__ == "__main__":
#     main()