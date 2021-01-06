import numpy as np
from math import cos, sin, pi, sqrt, pow, log10
import pymap_elites.map_elites as map_elites

class Arm2d:
  def __init__(self, lengths):
    self.n_dofs = len(lengths)
    self.lengths = np.concatenate(([0], lengths))
    self.joint_xy = []

  def fw_kinematics(self, p):
    assert(len(p) == self.n_dofs)
    p = np.append(p, 0)
    self.joint_xy = []
    mat = np.matrix(np.identity(4))
    for i in range(0, self.n_dofs + 1):
      m = [[cos(p[i]), -sin(p[i]), 0, self.lengths[i]],
           [sin(p[i]),  cos(p[i]), 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]]
      mat = mat * np.matrix(m)
      v = mat * np.matrix([0, 0, 0, 1]).transpose()
      self.joint_xy += [v[0:2].A.flatten()]
    return self.joint_xy[self.n_dofs], self.joint_xy

class Arm:
  def __init__(self, n_dof):
    self.x_dims = n_dof
    self.desc_length = 2

    # MAP-Elites Parameters
    self.order_of_magnitude = n_dof
    params = map_elites.default_params
    params["min"] = [-1.] * self.x_dims
    params["max"] = [1] * self.x_dims
    params["parallel"]    = True
    params["sigma_iso"]   = 0.003

    self.params = params

  def express(self, xx):
    return xx

  def randomInd(self, dim_x, params):
    return np.random.randn(dim_x)

  def random_vae_ind(self, dim_z, params):
    z = np.random.randn(dim_z)
    return z

def evaluate(xx):
  lengths = [1./xx.shape[0]] * xx.shape[0]
  arm = Arm2d(lengths)
  x = xx * pi / 2 # rescaling min/max 
  p, _ = arm.fw_kinematics(x)
  regularity = -np.std(x)
  return regularity, p / 2.0 + 0.5 # p in [-1,1]