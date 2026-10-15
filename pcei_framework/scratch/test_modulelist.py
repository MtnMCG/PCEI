import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2)])

m = MyModel()
m.layers.insert(1, nn.Linear(2, 2))
print("state_dict keys:", m.state_dict().keys())
