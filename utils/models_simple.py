import torch.nn as nn
import torch.functional as F

class MLP(nn.Module):
    """Model with input size (-1, 28, 28) for MNIST 10-classes dataset."""

    def __init__(self, input_size=2, output_size=3, hidden_units=10):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_units)
        self.fc = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.fc(x)
        return x

class MyLinear(nn.Module):
    """Model with input size (-1, 28, 28) for MNIST 10-classes dataset."""


    def __init__(self, input_size=2, output_size=3):
        super(MyLinear, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x