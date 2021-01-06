import numpy
import math
import numpy as np

class ActionController():
    def __init__(self, actions, save=False):
        assert(actions.shape[1] == 12)
        self.actions = actions
        self.i = 0
        self.state = np.zeros(18)
        self.k = 0
        self.save = save
        if self.save:
            np.save('actions.npy', actions)
            self.all_states_real = []
            self.all_states_target = []

    def step(self, simu):
        self.state = simu.get_joints_positions()
        if self.save:
            self.all_states_real += [self.state] 

        j = 0
        k = self.k % self.actions.shape[0]
        for i in range(0, 18, 3):
            self.state[i    ] = self.actions[k, j    ] + self.state[i    ]
            self.state[i + 1] = self.actions[k, j + 1] + self.state[i + 1]
            self.state[i + 2] = self.actions[k, j + 1] + self.state[i + 2]
            j += 2
        self.state = np.clip(self.state, a_min=-math.pi/4, a_max=math.pi/4)
        self.k += 1
        if (self.save):
            self.all_states_target += [self.state] 
            np.save('states_real.npy', np.array(self.all_states_real))
            np.save('states_targets.npy', np.array(self.all_states_target))
        
        return self.state