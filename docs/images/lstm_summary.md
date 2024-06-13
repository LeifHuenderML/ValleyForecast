# LSTM Class Documentation

## Overview

The `LSTM` class is a custom implementation of the LSTM (Long Short-Term Memory) network, which is a type of recurrent neural network used in deep learning. This class extends `torch.nn.Module` and is designed to process sequences of data. It includes several fully connected layers after the LSTM layers to facilitate additional processing and learning from the LSTM outputs.

## Class Definition

```python
class LSTM(nn.Module):

This class inherits from PyTorch's nn.Module and includes an LSTM layer followed by three linear (fully connected) layers.
Constructor Parameters

    input_size (int, default=19): The number of expected features in the input x.
    hidden_size (int, default=128): The number of features in the hidden state h.
    num_layers (int, default=1): Number of recurrent layers.
    bias (bool, default=True): If False, then the layer does not use bias weights.
    batch_first (bool, default=True): If True, then the input and output tensors are provided as (batch, seq, feature).
    dropout (float, default=0.1): If non-zero, introduces a dropout layer on the outputs of each layer except the last layer.
    bidirectional (bool, default=False): If True, becomes a bidirectional LSTM.
    proj_size (int, default=0): If greater than 0, the LSTM will have an additional linear layer of size proj_size at the output of each time step.

Attributes

    lstm: The LSTM network configured with the specified parameters.
    fc1, fc2, fc3: Linear layers for further processing after LSTM layer outputs.
    relu: Activation function.
    dropout1, dropout2: Dropout layers for regularization.
    device: Device configuration (either CPU or GPU) depending on CUDA availability.

Methods
forward(self, x)

Defines the forward pass of the LSTM module.
Parameters:

    x (Tensor): The input sequence data.

Returns:

    out (Tensor): The output after processing the input sequence through the LSTM and linear layers.

Details:

The method initializes hidden states, moves them to the appropriate device, processes the input through the LSTM, and then sequentially through the linear layers and activation functions, applying dropout as configured.
Usage Example

