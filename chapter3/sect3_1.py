import math
import time
import numpy as np
import torch

from d2l import torch as d2l

if __name__ == '__main__':
    c = torch.zeros(1024)
    a = torch.ones(1024)
    b = torch.ones(1024)

    timer = d2l.Timer()
    for i in range(1000):
        for n in range(1024):
            c[n] = a[n] + b[n]

    print(f'{timer.stop():.5f} sec')

    timer.start()
    for i in range(1000):
        d = a + b
    print(f'{timer.stop():.5f} sec')