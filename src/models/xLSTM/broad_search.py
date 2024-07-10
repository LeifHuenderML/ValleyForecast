from xlstm import GridSearch
import time
import torch
from torch.utils.data import DataLoader

def main():
    train = torch.load('/home/intellect/Documents/Research/Current/ValleyForecast/data/cleaned/train.pt')
    val = torch.load('/home/intellect/Documents/Research/Current/ValleyForecast/data/cleaned/val.pt')
    batch_size = 32
    train_loader = DataLoader(train, batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=True, drop_last=True)

    param_dict = {
        'hidden_size': [64, 128, 256, 512],
        'stack_config': ['ss', 'ms', 'mm', 'sm', 'ssm', 'sms', 'mss', 'smm', 'mmm', 'mms', 'mss'],
        'dropout': [0.0, 0.1, 0.15, 0.2],
    }
    grid_search = GridSearch(train_loader, val_loader)
    grid_search.search(param_dict)

if __name__ == "__main__":
    main()

