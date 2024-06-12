import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
import plotly.express as px
from itertools import product


current_script_path = Path(__file__).resolve()
nord_module_path = current_script_path.parents[2] / 'graphvis'
sys.path.insert(0, str(nord_module_path))
from nord import Nord


class LSTM(nn.Module):   
    def __init__(self, input_size=19, hidden_size=128, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False, proj_size=0):

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
        self.fc = nn.Linear(hidden_size, 1)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):   
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()

        h0 = h0.to(self.device)
        c0 = c0.to(self.device)

        _, (hn, _) = self.lstm(x,(h0,c0))
        out = self.fc(hn[0]).flatten()

        return out
    
class Trainer():
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer): 
        self.model = model.to(model.device)
        self.train_loader = train_loader   
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = model.device
        self.train_losses = []
        self.val_losses = []
        self.patience_counter = 0


    def train_one_epoch(self,):
        num_batches = len(self.train_loader)
        total_loss = 0
        self.model.train()
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
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
            #end training early if the model stops improving
            if self.early_stopping():
                return self.model, self.train_losses, self.val_losses
            if epoch % 10 == 0:
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
    
    def early_stopping(self, patience=20, delta=1):
        if len(self.val_losses) >= 2:
            if (self.val_losses[-1] + delta) >  self.val_losses[-2]:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    return True
            else:
                self.patience_counter = 0
        return False           

class GridSearch():
    def __init__(self, train_loader, val_loader, loss_fn, optimizer):
        self.train_loader = train_loader
        self.val_loader =val_loader
        self.loss_fn =loss_fn
        self.optimizer = optimizer
        self.train_loss_df = pd.DataFrame()
        self.val_loss_df = pd.DataFrame()


    def search(self, param_dict):
        keys = param_dict.keys()
        values = param_dict.values()
    
        for combo in product(*values):
            param_combo = dict(zip(keys, combo))
            train_losses, val_losses = self.evaluate_model(**param_combo)
            self.train_loss_df.concat([self.train_loss_df, train_losses])
            self.val_loss_df.cat([self.val_loss_df, val_losses])

    def evaluate_model(self, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size):
        model = LSTM(hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size)
        trainer = Trainer(model, self.train_loader, self.val_loader, self.loss_fn, self.optimizer)
        _, train_losses, val_losses = trainer.train()
        return train_losses, val_losses
    

