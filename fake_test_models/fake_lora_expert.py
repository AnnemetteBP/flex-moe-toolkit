import torch
import torch.nn as nn



class FakeLoRAExpert(nn.Module):
    def __init__(self, hidden_size, rank=4):
        super().__init__()
        self.A = nn.Linear(hidden_size, rank, bias=False)
        self.B = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, x):
        return x + self.B(self.A(x))    

"""
Test fake lora expert:

class FakeLoRAExpert(nn.Module):
    def __init__(self, hidden_size, rank=4):
        super().__init__()
        self.A = nn.Linear(hidden_size, rank, bias=False)
        self.B = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, x):
        return x + self.B(self.A(x))
"""