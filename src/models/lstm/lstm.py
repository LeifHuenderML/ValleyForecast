"""
 * @file lstm.py
 * @author Leif Huender
 * @brief contains the class for the LSTM, lstm Trainer, and the GridSearch, 
    used as a baseline model to compare performance up against the xlstm
 * @version 0.1
 * @date 2024-06-12
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
This class is a customized LSTM network built upon PyTorch's nn.Module. It features an LSTM layer followed by 
three linear (fully connected) layers to process sequential data effectively. The class is designed for tasks that 
require handling sequences, such as time series prediction or natural language processing.

Constructor Parameters:
    input_size (int, default=19): The number of expected features in the input x.
    hidden_size (int, default=128): The number of features in the hidden state h.
    num_layers (int, default=1): Number of recurrent layers.
    bias (bool, default=True): If False, the layer does not use bias weights.
    batch_first (bool, default=True): If True, then the input and output tensors are provided as (batch, seq, feature).
    dropout (float, default=0.1): If non-zero, introduces a dropout layer on the outputs of each layer except the last layer.
    bidirectional (bool, default=False): If True, creates a bidirectional LSTM.
    proj_size (int, default=0): If greater than 0, adds a linear layer of this size at the output of each time step.

Attributes:
    lstm (nn.LSTM): Configured LSTM network based on the specified parameters.
    fc1, fc2, fc3 (nn.Linear): Linear layers for additional processing after the LSTM output.
    relu (nn.ReLU): Activation function used after linear layers.
    dropout1, dropout2 (nn.Dropout): Dropout layers placed after linear layers to prevent overfitting.
    device (str): Device configuration (CPU or GPU) based on CUDA availability.

Methods:
    forward(self, x):
        Processes the input sequence through the LSTM and linear layers, applying activations and dropouts as configured.

        Parameters:
            x (Tensor): The input sequence data.

        Returns:
            out (Tensor): The final output of the network after processing.

Usage Example:
    # Assuming `model` is an instance of LSTM
    input_data = torch.randn(10, 5, 19)  # Example dimensions: batch_size=10, sequence_length=5, input_features=19
    output = model(input_data)
    print(output)

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
The Trainer class manages the training and validation of PyTorch models. It is responsible for executing the training loop,
handling batch processing, loss computations, and epoch management. The class also implements early stopping based on validation loss.

Attributes:
    model (torch.nn.Module): The model being trained.
    train_loader (DataLoader): DataLoader for the training dataset.
    val_loader (DataLoader): DataLoader for the validation dataset.
    loss_fn (function): Loss function to measure the model's performance.
    optimizer (Optimizer): Optimizer for updating model's weights.
    device (str): Device on which the model is trained (CPU or GPU).
    train_losses (list): Records the average training loss for each epoch.
    val_losses (list): Records the average validation loss for each epoch.
    patience_counter (int): Tracks the number of epochs with no improvement.

Methods:
    train_one_epoch(): Trains the model for one epoch and returns the average loss.
    validate_one_epoch(): Validates the model for one epoch and returns the average loss.
    train(epochs): Conducts the training and validation process for a given number of epochs.
    predict(test_loader): Predicts the output for the given test dataset.
    plot(): Visualizes the training and validation losses over each epoch.
    create_loss_df(): Creates a DataFrame with training and validation loss data.
    early_stopping(patience, delta): Checks if training should be stopped early based on validation performance.

Usage Example:
    model = MyModel()
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer)
    trained_model, train_losses, val_losses = trainer.train(epochs=10)
    predictions = trainer.predict(DataLoader(test_dataset, batch_size=32))
    trainer.plot()
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
    
