import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import plotly.express as px
from itertools import product
from torch.utils.data import DataLoader

current_script_path = Path(__file__).resolve()
nord_module_path = current_script_path.parents[2] / 'graphvis'
sys.path.insert(0, str(nord_module_path))
from nord import Nord

'''
Standard lstm from pytorch with 3 linear layers leading to a regressor value.
'''
class LSTM(nn.Module):   
    def __init__(self, input_size=19, hidden_size=128, num_layers=1, bias=True, batch_first=True, dropout=0.1, bidirectional=False, proj_size=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), int(hidden_size/4))
        self.fc3 = nn.Linear(int(hidden_size/4), 1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)  
        self.dropout2 = nn.Dropout(dropout)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):   
        #create hidden states
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        #put them on gpu
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        # run data data through the forward pass
        _, (hn, _) = self.lstm(x,(h0,c0))
        out = self.fc1(hn[0])
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = out.flatten()
        return out

'''
Trains a model on trainging and validation data accross n epochs
'''
class Trainer():
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer): 
        self.model = model.to(model.device) #transfer the model to the gpu
        self.train_loader = train_loader   
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = model.device #define the device for the training data to go on to
        self.train_losses = [] #stores the avg loss of the training for each epoch
        self.val_losses = [] #stores the avg loss for the test set for each epoch
        self.patience_counter = 0 #counts how many times the model does not pass the performanch threshould between epochs


    def train_one_epoch(self,):
        #get the length for calcualting the average loss at the end
        num_batches = len(self.train_loader)
        total_loss = 0

        self.model.train()
        for x, y in self.train_loader:
            # move data to the same devise as the model
            x, y = x.to(self.device), y.to(self.device)
            # inferencce
            output = self.model(x)
            loss = self.loss_fn(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss/num_batches

        return avg_loss

    def validate_one_epoch(self,):
        num_batches = len(self.val_loader)
        total_loss = 0

        self.model.eval()
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                total_loss += self.loss_fn(output, y).item()
                
        avg_loss = total_loss/num_batches
        return avg_loss

    def train(self, epochs):
        for epoch in range(1, epochs+1):
            #train and validate a batch
            train_loss = self.train_one_epoch()
            val_loss = self.validate_one_epoch()
            #calculate the RMSE
            train_loss = np.sqrt(train_loss)
            val_loss = np.sqrt(val_loss)
            #add them to the list that keeps track of losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            # end training early if the model stops improving
            if self.early_stopping():
                return self.model, self.train_losses, self.val_losses
            if epoch % 1 == 0:
                print(f'Epoch: {epoch}/{epochs}\n---------')
                print(f'Train Loss RMSE: {train_loss}, Validation Loss RMSE: {val_loss}')
                print()
        return self.model, self.train_losses, self.val_losses
        

    def predict(self, test_loader):
        output = torch.tensor([])
        self.model.eval()
        with torch.no_grad():
            for x, _ in test_loader:
                x= x.to(self.device)
                y_star = self.model(x)
                output = torch.cat((output, y_star), dim=0)
        return output
    
    def plot(self):
        nord_pallette = Nord()
        df = self.create_loss_df()
        fig = px.line(df, x='Epochs', y=['Train', 'Validation'], color_discrete_sequence=[nord_pallette.red, nord_pallette.dark_blue])
        fig.update_layout(
            title="Model RMSE",
            title_x = 0.5, 
            xaxis_title='Epochs',
            yaxis_title='Loss',
            legend_title='Data',
            paper_bgcolor=nord_pallette.white,
            plot_bgcolor=nord_pallette.light_grey,
        )
        fig.show()

    def create_loss_df(self):
        data = {
            'Epochs' : [i for i in range(len(self.train_losses))],
            'Train' : self.train_losses,
            'Validation' : self.val_losses
        }
        df = pd.DataFrame(data)
        return df
    
    def early_stopping(self, patience=5, delta=1):
        if len(self.val_losses) >= 2:
            if (self.val_losses[-1] + delta) >  self.val_losses[-2]:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    return True
            else:
                self.patience_counter = 0
        return False           

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
        current_min = float('inf')
        best_param_combo = ''
        # creates the cartesian product of all the combinations for searching over then loopss through them
        for combo in product(*values):
            param_combo = dict(zip(keys, combo))
            train_losses, val_losses = self.evaluate_model(**param_combo)
            self.train_loss_df = pd.concat([self.train_loss_df, pd.Series(train_losses)])
            self.val_loss_df = pd.concat([self.val_loss_df, pd.Series(val_losses)])
            if min(train_losses) < current_min: 
                curent_min = min(train_losses)
                best_param_combo = param_combo
        
        return current_min, best_param_combo

    def evaluate_model(self, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size):
        print( hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size)
        model = LSTM(input_size=19, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, proj_size=proj_size)
        loss_fn = nn.MSELoss()
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,)

        trainer = Trainer(model, self.train_loader, self.val_loader, loss_fn, optimizer=optimizer)
        _, train_losses, val_losses = trainer.train(epochs=1000)
        return train_losses, val_losses
    
