import sys
import numpy as np
import pymap_elites.map_elites as map_elites
from sklearn.neighbors import KDTree
from vae_map_elites import vae_map_elites
from vae import *


def main():
# -- Parse input arguments ----------------------------------------------------#
# python3 dde.py arm20  map 1500                   # map-elites w/new cvt-archive
# python3 dde.py arm20  map maps/ring_6.dat        # map-elites w/predefined archive
# python3 dde.py arm20  vae maps/ring_6.dat 10     # dde-elites w/10 latent dimensions
# python3 dde.py arm200 vae maps/ring_6.dat 32     # dde-elites w/32 latent dimensions
# python3 dde.py arm20  dde maps/ring_6.dat dde/dde_arm20.pt # map-elites dde representation

  domain  = sys.argv[1] # arm | hex
  mode    = sys.argv[2] # map | vae | dde
  archive = sys.argv[3] # centroid_file | n_centroids

  if mode == 'map':
    print('\n[****] Running MAP-Elites [****]')

  elif mode == 'line':
    print('\n[****] Running MAP-Elites w/Line Search [****]')    

  elif mode == 'vae_only':
    print('\n[****] Running VAE-Elites using only VAE [****]') 
    nZ  = int(sys.argv[4]) # number of latent dimensions
  
  elif mode == 'vae':
    print('\n[****] Running VAE-Elites [****]')
    nZ  = int(sys.argv[4]) # number of latent dimensions
  
  elif mode == 'vae_line':
    print('\n[****] Running VAE-Elites w/line mutation[****]')
    nZ  = int(sys.argv[4]) # number of latent dimensions        
  
  elif mode == 'dde':
    print('\n[****] Running DDE-Elites [****]')
    vaeFile = sys.argv[4]  # .pt file of pretrained VAE    
  
  else:
    print('Invalid mode selected (map/vae/dde)')
    exit(1)

# -- Setup --------------------------------------------------------------------#
  # Load or Create Archive (number of niches or pre-defined centroids)
  if archive.endswith('.dat'): # Centroid file
    centroids = np.loadtxt(archive) #centroids = np.loadtxt('maps/ring_6.dat')
    n_niches  = np.shape(centroids)[0]
  else:
    centroids = np.empty(shape=(0,0));
    n_niches = int(archive)

  # Set Domain
  if domain[0:3] == 'arm':
    from domain.arm.planarArm import Arm2d, Arm, evaluate
    n_joints = int(domain[3:])
    print("Number of Joints", n_joints)
    d = Arm(n_joints) # Numbers after arm is number of joints
    if n_joints > 200:
      d.params["random_init"] = 0.01
    else:
      d.params["random_init"] = 0.05
    n_gen = 10000


  elif domain == 'hex':
    from domain.hexa.hexapod import Hex, evaluate
    d = Hex()
    d.params["random_init"] = 0.05;
    n_gen = 50000

  else:
    print('Invalid Domain (e.g. hex/arm20/arm200)')
    exit(1)

  x_dims                = d.x_dims
  desc_length           = d.desc_length
  params                = d.params
  params["gen_to_phen"] = d.express
  params["random"]      = d.randomInd
  params["batch_size"]  = 100
  params["trainMod"]    = 1   # number of gens between VAE training
  params["trainEpoch"]  = 5
  params["banditWindow"]= 1000
  params["vector_variation"] = True

  
  n_gen+= 1

  # Set Logging Hyperparameters
  params["dump_period"] = 100
  params["save_format"] = 'bin'
  params["print_mod"]   = 5
  log_file = open('map_elites_log.dat', 'w+')
  vae_log  = open('vae_log.dat', 'w+')


  # Tiny run testing
  #params["random_init"] = 5./float(n_niches);  
  #n_gen = 3

# -- Test Algorithms ----------------------------------------------------------#
  if mode == 'map':
    params["sigma_line"] = 0.0
    params["vector_variation"] = False
    map_elites.compute(d.desc_length, d.x_dims, evaluate, params=params,
                 centroids=centroids, n_niches=n_niches, n_gen=n_gen, 
                 log_file=log_file)

  if mode == 'line':
    params["sigma_line"] = 0.1
    params["vector_variation"] = False    
    map_elites.compute(d.desc_length, d.x_dims, evaluate, params=params,
                 centroids=centroids, n_niches=n_niches, n_gen=n_gen, 
                 log_file=log_file)


  if mode == 'vae':
    params["bandit_prob_xover"] = [0, 0.25, 0.5, .75, 1.0]  
    params["bandit_line_sigma"] = [0.0]
    vae_map_elites(d.desc_length, d.x_dims, evaluate, params,
                centroids=centroids, n_niches=n_niches, n_gen=n_gen, 
                model=VecVAE, latent_length=nZ, vae_log=vae_log,
                log_file=log_file)

  if mode == 'vae_line':
    params["bandit_prob_xover"] = [0, 0.25, 0.5, .75, 1.0] 
    #params["bandit_prob_xover"] = [0, 1.0]  
    params["bandit_line_sigma"] = [0.1, 0.0]  
    
    vae_map_elites(d.desc_length, d.x_dims, evaluate, params,
                centroids=centroids, n_niches=n_niches, n_gen=n_gen, 
                model=VecVAE, latent_length=nZ, vae_log=vae_log,
                log_file=log_file)

  if mode == 'vae_only':
    params["bandit_prob_xover"] = [1.0] 
    params["bandit_line_sigma"] = [0.0] # this is never used
    vae_map_elites(d.desc_length, d.x_dims, evaluate, params,
                centroids=centroids, n_niches=n_niches, n_gen=n_gen, 
                model=VecVAE, latent_length=nZ, vae_log=vae_log,
                log_file=log_file)


# -- Use Data-Driven Encoding -------------------------------------------------#
  if mode == 'dde':
    print('\n[**] Loading Data Driven Encoding from: ', sys.argv[4], ' [**]')
    weights = torch.load(sys.argv[4])
    z_dims = list(weights()['fc21.bias'].size())[0] # Decoder input
    x_dims = list(weights()['fc4.bias'].size())[0]  # Decoder output
    print('\n[*] Using ', str(x_dims), 'D Encoding with ', str(z_dims), ' Latent Dimensions [*]')

    n_gen = int(n_gen/10)+1
    params["dump_period"] = 100

    # Mutation Parameters
    params["min"]         = [-5.0] * z_dims
    params["max"]         = [ 5.0] * z_dims
    params['sigma_line']  =   0.1 # was 0.0 in original experiments...
    params["sigma_iso"]   =   0.015
    params["random"] = d.random_vae_ind
    params["vector_variation"] = False


    # Load DDE
    vae = VecVAE(x_dims,z_dims)
    vae.load_state_dict(weights())
    params["gen_to_phen"] = vae.express

    print('\n[**] Optimizing DDE with MAP-Elites [**]')
    map_elites.compute(d.desc_length, z_dims, evaluate, params=params,
                 centroids=centroids, n_niches=n_niches, n_gen=n_gen, 
                 log_file=log_file)

if __name__ == '__main__':
  main()



