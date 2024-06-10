import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, random_split

# combine the 2 datasets 

print('Loading ../../data/original/rates.xlsx')
df = pd.read_excel('../../data/original/rates.xlsx')
