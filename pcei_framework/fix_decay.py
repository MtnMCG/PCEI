import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.engine.physics import EntanglementBroker, ResonanceDetector

broker = EntanglementBroker()
broker.active_bonds[('a', 'b')] = 0.5
broker.tick_decay()
print(broker.active_bonds)

