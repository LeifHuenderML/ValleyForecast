"""
 * @file ml.py
 * @author Leif Huender
 * @brief a suite of ml functions crafted by leif
 * @version 0.1
 * @date 2024-06-17
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
from numpy import exp
import torch
import time

def sigmoid(input):
    return 1/ (1 + (exp(-input))) 

def tanh(input):
    return (exp(input) - exp(-input)) / (exp(input) + exp(-input)) 
    
def mm(input, mat2):
    '''
    If input is a (n by m) tensor, mat2 is a (m by p) tensor, out will be a (n by p) tensor.
    Does not broadcast.
    '''
    if input.shape[1] != mat2.shape[0]:
        raise ValueError('Number of columns in input must be the same as the number of rows in mat2')

    out = torch.zeros(input.shape[0], mat2.shape[1])

    for i in range(input.shape[0]): 
        for j in range(mat2.shape[1]):  
            for k in range(input.shape[1]): 
                out[i][j] += input[i][k] * mat2[k][j]
    return out
    


def main():
    #tests all the functions for ml
    mat1 = torch.randn(300,500)
    mat2 = torch.randn(500,300)
    # print(f'Original Tensor: {mat1}')
    # print(f'Sigmoid {sigmoid(mat1)}')
    # print(f'Tanh {tanh(mat1)}')
    # print(f'Torch mm: {torch.mm(mat1, mat2)}, \nML mm: {mm(mat1, mat2)}')
    start = time.time()
    torch.mm(mat1, mat2)
    end = time.time()
    print(f'Torch MM time {end-start}')
    start = time.time()
    mm(mat1, mat2)
    end = time.time()
    print(f'ML MM time {end-start}')
    

if __name__ == '__main__':
    main()