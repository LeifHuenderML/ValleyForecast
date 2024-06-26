import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, ):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_weights = nn.Linear(input_size, hidden_size * 4,device='cuda')
        self.hidden_weights = nn.Linear(hidden_size, hidden_size*4,device='cuda')

        self.reset_parameters()

    def reset_parameters(self,):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def init_hidden(self, input):
        hidden = Variable(input.new_zeros(input.size(0), self.hidden_size)).to('cuda')
        return (hidden, hidden, hidden, hidden)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input)
        h_t_p, c_t_p, n_t_p, m_t_p = hidden
        
        gates = self.input_weights(input) + self.hidden_weights(h_t_p)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
        
        z_t = torch.tanh(cell_gate)
        i_t = torch.exp(input_gate)  
        f_t = torch.exp(forget_gate)  
        o_t = torch.sigmoid(output_gate)
        
        m_t = torch.max(torch.log(f_t) + m_t_p, torch.log(i_t))  
        
        i_t_prime = torch.exp(input_gate - m_t)  
        f_t_prime = torch.exp(forget_gate + m_t_p - m_t)  
        
        c_t = f_t_prime * c_t_p + i_t_prime * z_t
        n_t = f_t_prime * n_t_p + i_t_prime
        h_t = o_t * (c_t / n_t)
        
        return (h_t, c_t, n_t, m_t)
    
class sLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.slstm = sLSTMCell(input_size, hidden_size)

    def forward(self, input, hidden=None):
        outputs = []
        for time_step in range(input.size(1)):
            hidden = self.slstm(input[:,time_step,:], hidden)
            outputs.append(hidden)
        return outputs, hidden

class sLSTM(nn.Module):
    def __init__(self, input_size=19, hidden_size=256,):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.slstm = sLSTMLayer(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc3 = nn.Linear(hidden_size//4, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        out, hidden = self.slstm(input)
        out = self.fc1(hidden[0])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = out.squeeze()
        return out

class sLSTMTrainer:
    def __init__(self, loader):
        super(sLSTMTrainer, self).__init__()
        self.model = sLSTM().to('cuda')
        self.loader = loader
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epochs):
        losses = []
        self.model.train()
        for epoch in range(1, epochs+1):
            num_batches = len(self.loader)
            total_loss = 0
            for x, y in self.loader:
                x, y = x.to('cuda'), y.to('cuda')
                output = self.model(x)
                loss = self.loss_fn(output, y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
            avg_loss = total_loss/num_batches
            avg_loss = np.sqrt(avg_loss)
            losses.append(avg_loss)
            print(f'Epoch {epoch} RMSE Loss: {avg_loss}')
        return losses
    


def main():
    train = torch.load('/home/intellect/Documents/Research/Current/ValleyForecast/data/cleaned/train.pt')
    batch_size = 32
    train_loader = DataLoader(train, batch_size, shuffle=True, drop_last=True)

    trainer = sLSTMTrainer(train_loader)
    trainer.train(100)

if __name__ == "__main__":
    main()