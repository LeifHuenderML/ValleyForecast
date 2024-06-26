import torch
from slstm import sLSTMTrainer
from torch.utils.data import DataLoader



def main():
    train = torch.load('/home/intellect/Documents/Research/Current/ValleyForecast/data/cleaned/train.pt')
    batch_size = 32
    train_loader = DataLoader(train, batch_size, shuffle=True, drop_last=True)

    trainer = sLSTMTrainer(train_loader)
    trainer.train(100)

if __name__ == "__main__":
    main()