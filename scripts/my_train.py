import os
import sys
import json
import torch
import random

import numpy as np
import torch.optim as optim

sys.path.append("../utils")  # HACK add the lib folder
sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder

from models.instancerefer import InstanceRefer


if __name__ == "__main__":
    net = InstanceRefer(7).cuda()
    out = net(None)
