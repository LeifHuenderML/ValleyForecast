import torch
from torchinfo import summary
from torchview import draw_graph

class Trainer:
    def __init__(self, model, loader,):
        self.model = model
        self.loader = loader
        self.losses = []
    def train(self, epochs, early_stopping=True, model_checkpoints=True):
        # print(f'Training {self.model}')
        
        # for epoch in range(1, epochs+1):
        return self.losses