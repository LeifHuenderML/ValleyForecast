from slstm import sLSTM
import torch

lstm = sLSTM(input_size=19, hidden_size=256,)

x = torch.randn((32,365,19)) 

lstm(x)

