import torch.nn.functional as F
import numpy as np
import torch as tr
from sklearn.model_selection import train_test_split


a = [
    [1,2,3,],
    [4,5,6,],
]

b = [
    [1,2,3,],
    [4,5,6,],
]

a,b,c,d=train_test_split(a, b, test_size=0.5)

print(a)