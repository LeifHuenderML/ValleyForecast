"""
 * @file fine_search.py
 * @author Leif Huender
 * @brief 
 * @version 0.1
 * @date 2024-06-13
 * This script runs a fine search narrowing down further the best performant model on a search space of 352 models 
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

import lstm
import torch
from torch.utils.data import DataLoader

def main():
    #load all the datasets
    train = torch.load('../../../data/cleaned/train.pt')
    val = torch.load('../../../data/cleaned/val.pt')
    test = torch.load('../../../data/cleaned/test.pt')

    #make them into dataloaders
    batch_size = 32
    train_loader = DataLoader(train, batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test, batch_size, shuffle=True, drop_last=True)
    grid_search = lstm.GridSearch(train_loader, val_loader)

    param_dict = {
        'hidden_size' : [200, 210, 220, 230, 240, 250, 256, 260, 270, 280, 290],
        'num_layers' : [2, 3, 4, 5],
        'bias' : [True, False],
        'batch_first' : [True],
        'dropout' : [0, 0.1, 0.2, 0.3],
        'bidirectional' : [False],
        'proj_size' : [0]
    }

    grid_search.search(param_dict)

if __name__ == '__main__':
    main()