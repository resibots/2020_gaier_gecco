import sys
import numpy as np
import pymap_elites.map_elites as map_elites
import pyhexapod.simulator as simulator
import time
import math
import pybullet
import multiprocessing
from action_controller import ActionController
import tqdm # progress bar
 

def extract_states(xx, id, freq=20, dt=1./240., total_length=3):
    simu = simulator.HexapodSimulator(gui=False, dt = dt, control_dt=1./freq, urdf='pyhexapod/urdf/pexod.urdf')
    control_period = simu.control_period
    actions = xx.reshape(int(xx.shape[0] / 12), 12) * 0.25
    controller = ActionController(actions)# one update every 50 steps
    joints = np.arange(0, 18)
    uncontrolled_joints =  joints[2:len(joints):3]
    c_joints = np.setdiff1d(joints, uncontrolled_joints)
    assert(c_joints.shape[0] == 12)

    steps =  int(total_length / simu.dt)
    # state = [joint_pos [12], angular_pos [3], contacts [6]] => 21
    # action = 12  => 33
    # id => 34
    # number => 35
    # expected distance => 36
    states = []
    fit = -1e10
    dead = False
    i = 0
    joint_pos_array=np.zeros((steps, 12))
    while i < steps and not dead:
        # log
        joint_pos = simu.get_joints_positions()[c_joints]
        joint_pos_array[i, :] = joint_pos
        if i % control_period == 0:
            pos = simu.get_pos()
            angles = pybullet.getEulerFromQuaternion(pos[1])
            contacts = np.ones(6)
            for l in range(0, 6):
                if len(simu.descriptor[simu.leg_link_ids[l]]) > 0: 
                    contacts[l] = simu.descriptor[simu.leg_link_ids[l]][-1]
            s = np.zeros(36)##33
            s[0:12 ] = joint_pos
            s[12:15] = angles
            s[15:21] = contacts
            s[21:33] = actions[len(states)]
            s[33] = id
            s[34] = len(states)
            s[35] = pos[0][0]
            states += [s]

        # simu
        simu.step(controller)
        p = simu.get_pos()[0] 
        a = pybullet.getEulerFromQuaternion(simu.get_pos()[1])
        out_of_corridor = abs(p[1]) > 0.5
        out_of_angles = abs(a[0]) > math.pi/8 or abs(a[1]) > math.pi/8 or abs(a[2]) > math.pi/8
        if out_of_angles or out_of_corridor:
            dead = True
        i += 1
    simu.destroy()
    return states


def flatten(arr):
  for i in arr:
    if isinstance(i, list):
      yield from flatten(i)
    else:
      yield i

def evaluate(x):
    state,i = x
    s = extract_states(state, i)
    return s

def main(filename):
    np.set_printoptions(suppress=True)

    print('loading', filename)
    a = np.load(filename)
    print('sorting...')
    archive = a[(-a[:,0]).argsort()]
    i_max = np.argmax(np.array(archive[:,0]))
    mean = np.mean(np.ma.masked_values(archive[:,0], -1e10))
    print("MEAN fit:", mean)
    print("fit=", archive[i_max, 0])
    print(archive[0,0])
    desc = 360
    states = []
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    to_evaluate = []
    k = archive.shape[0]
    for i in range(0, k):
        x = archive[i, desc+1:archive.shape[1]]
        to_evaluate += [(x, i)]
    # states = pool.map(evaluate, to_evaluate)
    # => version with progress bar!
    states = list(tqdm.tqdm(pool.imap(evaluate, to_evaluate), total=len(to_evaluate)))
    #states = list(tqdm.tqdm(map(evaluate, to_evaluate), total=len(to_evaluate)))
    #np.save('joints.npy', np.array(states))
    final_states = np.array(list(flatten(states)))

    #print(final_states)
    np.save("states.npy", final_states)

main(sys.argv[1])