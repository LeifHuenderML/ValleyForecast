import torch
import torch.nn as nn

class sLSTM(nn.Module):
    def __init__(self,input_size=19, hidden_size=365):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTMCell(input_size,hidden_size)

        self.exp_forget_gate = nn.Linear(hidden_size,hidden_size)
        self.exp_input_gate = nn.Linear(hidden_size,hidden_size)

        self.reset_parameters()
    
    def reset_parameters(self,):
        nn.init.xavier_uniform_(self.lstm.weight_ih)
        nn.init.xavier_uniform_(self.lstm.weight_hh)
        nn.init.zeros_(self.lstm.bias_ih)
        nn.init.zeros_(self.lstm.bias_hh)
        nn.init.xavier_uniform_(self.exp_forget_gate.weight)
        nn.init.zeros_(self.exp_forget_gate.bias)
        nn.init.xavier_uniform_(self.exp_input_gate.weight)
        nn.init.zeros_(self.exp_input_gate.bias)

    def init_hidden(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_size, device=self.lstm.weight_ih.device)
        c = torch.zeros(batch_size, self.hidden_size, device=self.lstm.weight_ih.device)
        return (h,c)
    
    def forward(self, input, hidden=None):
        batch_size = input.size(0)
        seq_length = input.size(1)

        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        out = []
        for t in range(seq_length):
            x = input[:,t,:]
            h,c = self.lstm(x, hidden[0], hidden[1])

            f = torch.exp(self.exp_forget_gate(h))
            i = torch.exp(self.exp_input_gate(h))

            c = f * c + i * self.lstm.weight_hh.new_zeros(batch_size, self.hidden_size)

            x = h
            hidden = (h,c)
            out.append(x)
        out = torch.stack(out, dim=1)
        return out, hidden


