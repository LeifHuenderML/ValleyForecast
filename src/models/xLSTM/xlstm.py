import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from mLSTM.mlstm import mLSTM
from sLSTM.slstm import sLSTM

class xLSTM(nn.Module):
    def __init__(self,):
        o = mLSTM()
        n = sLSTM()
        