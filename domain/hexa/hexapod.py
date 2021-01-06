import sys
import numpy as np
import time
from math import pi

sys.path.insert(1, './domain/hexa/')
sys.path.insert(1, './')
sys.path.insert(1, './domain/hexa/pyhexapod')


import pymap_elites.map_elites as map_elites
import domain.hexa.pyhexapod.simulator as simulator
from   domain.hexa.pos_controller import PosController
import pybullet

class Hex:
  def __init__(self, freq=20, seconds=1, dofs=12):
    self.x_dims = freq * seconds * dofs
    self.desc_length = 2

    # MAP-Elites
    params = map_elites.default_params
    params["min"] = [-1.] * self.x_dims
    params["max"] = [1] * self.x_dims
    params["parallel"]    = True
    params["sigma_iso"]   = 0.003
    

    self.params = params

  def express(self, xx):
    return xx

  def randomInd(self, dim_x, params):
    x = np.zeros(dim_x)
    dofs = 12
    for i in range(dofs, dim_x, dofs):
      v = np.random.random(dofs) - 0.5
      x[i:i+dofs] = np.clip(x[i-dofs:i] + v, a_min=-1, a_max=1) 
    return x

  def random_vae_ind(self, dim_z, params):
    z = np.random.randn(dim_z)
    #x = params["gen_to_phen"](z)
    return z

def load(file_name, indx=[], fname="gait.mp4"):
  desc = 2
  archive = np.load(file_name)
  if not indx:
    indx = np.random.randint(0, archive.shape[0])
  print("Index: ", indx)
  mean = np.mean(np.ma.masked_values(archive[:,0], -1e10))
  print("MEAN fit:", mean)
  print("fit=", archive[indx, 0])
  print(archive.shape)
  genome = archive[indx, desc+1:archive.shape[1]]
  f, d = evaluate(genome, gui=True, fname=fname)
  print(d)
  print(archive[indx, 1:desc+1])
  print('fitness:', f)
  print(pybullet.getAPIVersion())


def evaluate(xx, gui=False, freq=20, dt=1./240., total_length=3, 
                      urdfPath='domain/hexa/pyhexapod/urdf/pexod.urdf', fname=[]):
  #urdfPath='../../../../domain/hexa/pyhexapod/urdf/pexod.urdf'
  urdfPath='/home/agaier2m/Code/vae-elites/domain/hexa/pyhexapod/urdf/pexod.urdf'
  #urdfPath='/nfs/hal01/jmouret/git/map_actions/pyhexapod/urdf/pexod.urdf'
  
  simu = simulator.HexapodSimulator(gui=gui, dt = dt, control_dt=1./freq, urdf=urdfPath)
  positions = xx.reshape(int(xx.shape[0] / 12), 12) * pi / 4# put in [-pi/4,pi/4]
  controller = PosController(positions)
  if gui:
    if fname:
      np.savetxt('positions.dat', positions)
      pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, fname)
  energy = None
  steps =  int(total_length / simu.dt)
  if gui:
    steps *= 1
  
  # simulate
  dead = False
  i = 0
  t0 = time.perf_counter()
  #print(steps)
  while i < steps and not dead:
    simu.step(controller)
    p = simu.get_pos()[0] 
    a = pybullet.getEulerFromQuaternion(simu.get_pos()[1])
    out_of_corridor = abs(p[1]) > 0.5
    out_of_angles = abs(a[0]) > pi/8 or abs(a[1]) > pi/8 or abs(a[2]) > pi/8
    if out_of_angles:
      energy = p[0]
      dead = True
    i += 1
    if(gui):
      time.sleep(0.005)
  #print(time.perf_counter() - t0, " ms")
  desc = np.array([simu.get_pos()[0][0], simu.get_pos()[0][1]]) / 2. + 0.5
  if dead:  
    fit = 0
  else:
    start = np.array([0.5,0.5])
    dist = np.linalg.norm(np.abs(desc-start), ord=2) # Euc dist is L2 norm
    energy = np.sum(np.abs(positions[1:len(positions)] - positions[0:len(positions)-1]))
    fit  = dist/energy
  # print('End Coord: ', desc)
  # print('Distance : ', dist)
  # print('Energy   : ', energy)
  # print('FIT      : ', fit, '\n')
  if gui:
    print("FIT:", fit)

  simu.destroy()# needed to avoid a memory leak!
  return fit, desc

def main():
  # * View Trial: [archive.npy] [ID] [outfile.mp4]
  print(sys.argv[1], int(sys.argv[2]) )
  if len(sys.argv) < 4:
    load(sys.argv[1], int(sys.argv[2]) )
  else:
    load(sys.argv[1], int(sys.argv[2]), sys.argv[3])


if __name__ == '__main__':
  main()



