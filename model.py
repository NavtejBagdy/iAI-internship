import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class CNN(nn.Module):
    def __init__(self, num_classes, vocab_size):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=2)
        self.fc = nn.Linear(256, num_classes)
        self.vocab_size = vocab_size

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.AdaptiveMaxPool1d(1)(x)
        x = x.squeeze(2)
        x = self.fc(x)
        return x


