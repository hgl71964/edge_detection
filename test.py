import torch.nn.functional as F
import numpy as np
import torch as tr
from sklearn.model_selection import train_test_split
import torch.nn as nn


l1 = nn.BCELoss()
l2 = nn.NLLLoss()

a = tr.tensor([
    [0.1],
    [0.9],
]).float().flatten()

b = tr.tensor([
    [0],
    [1],
]).float().flatten()

print(l1(a, b))

# print(l2(a,b))


print((tr.log(tr.tensor(0.9))+tr.log(tr.tensor(0.9)))/2)