import torch
import matplotlib.pyplot as plt
from SnakeGame import SnakeGame
from time import perf_counter_ns


# Settings
GRID_RES = (80, 80)
GRID_PIXEL = 10
MPS = 20

class SnakeGameAI (SnakeGame):

    def __init__(self):
        super(SnakeGameAI, self).__init__(GRID_RES, GRID_PIXEL, MPS)

        self.n_gmaes = 0
        self.epsilon = 0    # ramdomness
        self.gamma   = 0    # discount rate for deep Q Learning



if __name__ == '__main__':
    SnakeGameAI().run()