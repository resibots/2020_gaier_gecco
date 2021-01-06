import numpy
import math
import numpy as np

class PosController():
    def __init__(self, positions, save=False):
        assert(positions.shape[1] == 12)
        self.positions = positions
        self.i = 0
        self.k = 0
        self.save = save
        if self.save:
            np.save('positions.npy', positions)
       
         

    def step(self, simu):
        self.new_positions = np.zeros(18)
        k = self.k % self.positions.shape[0]
        j = 0
        for i in range(0, 18, 3):
            self.new_positions[i    ] = self.positions[k, j    ]
            self.new_positions[i + 1] = self.positions[k, j + 1] 
            self.new_positions[i + 2] = self.positions[k, j + 1] 
            j += 2
        self.k += 1
        return self.new_positions