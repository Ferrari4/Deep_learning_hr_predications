import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import os
import math
from torch.utils.data import TensorDataset, DataLoader

class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_lay, dropout_n):
        super(encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_lay = num_lay
        self.dropout = nn.Dropout(dropout_n)
        self.rnn = nn.LSTM(input_size, hidden_size, num_lay, dropout=dropout_n)

    def forward(self, x):
        _, (hidden, cell) = self.rnn(x)
        return hidden, cell

class decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_lay, dropout_n):
        super(decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_lay = num_lay
        self.dropout = nn.Dropout(dropout_n)
        self.rnn = nn.LSTM(input_size, hidden_size, num_lay, dropout=dropout_n)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        output, (hidden, cell) = self.rnn(x, (hidden, cell))
        predictions = self.fc(output).squeeze(0)
        return predictions, hidden, cell

class wrist_forcaster(nn.Module):
    def __init__(self, device):
        super(wrist_forcaster, self).__init__()
        self.encoder = encoder( input_size=3, 
                                hidden_size=128, 
                                num_lay=2, 
                                dropout_n=0.2)
        
        self.decoder = decoder( input_size=6,
                                hidden_size=128,
                                output_size=6,
                                num_lay=2,
                                dropout_n=0.5)
        self.device = device

    def forward(self, source, robot_future):
        batch_size = source.shape[1]
        target_len = robot_future.shape[0]
        outputs = source.new_zeros(target_len, batch_size, 6)
        hidden, cell = self.encoder(source)
        x = source[-1]
        for i in range(target_len):
            decoder_input = torch.cat((x, robot_future[i]), dim=-1)
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[i] = output
            x = output[:, :3]
        return outputs