import sys
import numpy as np
import cma
from vae import *
from cma.fitness_transformations import EvalParallel


# -- Target Behavior Descriptor Function ---------------------------------- -- #
def enc_direct(x):
  return x

def IK(xx, evaluate, target, encoding, verbose=False):
  fit, desc = evaluate(encoding(xx))
  if verbose:
    print('Descriptor:\t', desc, '\nTarget: \t',target, '\nOriginal Fit: \t',fit ); 
  return np.linalg.norm((desc-target), ord=2) # Euclidean distance to descriptor

def runCma(opts, n_dofs, target, evaluate, encoding):
  sigma = 0.3
  es = cma.CMAEvolutionStrategy(np.random.randn(n_dofs)*sigma, 0.1, opts) 
  es.restarts = np.inf
  log = []

  with EvalParallel(es.popsize + 1) as eval_all:
    while not es.stop():
      X = es.ask()
      es.tell(X, eval_all(IK, X, args=(evaluate,target,encoding)) )

      # CMA-ES loggers are a fucking abomination -- just record performance
      dataPt = np.array((es.countevals,es.best.f))
      if len(log)<1:
        log = dataPt
      else:
        log = np.c_[log,dataPt]

  old_fit, desc = evaluate(encoding(es.best.x_geno))
  n_eval = log[0,:]
  f_best = log[1,:]
  d_dist = log[1,-1]
  f_orig = old_fit
  x_best = es.best.x_geno
  return n_eval, f_best, d_dist, f_orig, x_best

def main():
# -- Parse input arguments ------------------------------------------------ -- #
# python3 cmaTarget.py arm20 target.csv             # hit targets w/standard rep
# python3 cmaTarget.py arm20 target.csv rep20.pt    # hit targets w/dde

  domain  = sys.argv[1] # arm20 | arm200 | hex
  target  = sys.argv[2] # .csv file containing 2D behvaior coordinates
  dde     = []

  if len(sys.argv) > 3:
    dde = sys.argv[3]   # .pt file containing pretrained ANN

# -- Setup ---------------------------------------------------------------- -- #
  # Load Targets and Domain
  target = np.loadtxt(target,delimiter=',')

  # Algorithm Hyperparams
  opts = cma.CMAOptions()  
  opts.set('BoundaryHandler', cma.BoundPenalty)
  opts.set('bounds',[-1,1])
  opts.set('verbose',-2)

  # Set Domain
  if domain[0:3] == 'arm':
    from domain.arm.planarArm import Arm2d, Arm, evaluate
    n_dofs = int(domain[3:])
    opts.set('maxfevals',5000)

  elif domain == 'hex':
    from domain.hexa.hexapod import Hex, evaluate
    n_dofs = 240
    opts.set('maxfevals',10000)

  else:
    print('Invalid Domain (e.g. hex/arm20/arm200)')
    exit(1)

  if dde:
    print('\n[**] Loading Data Driven Encoding from: ', dde, ' [**]')    
    
    weights = torch.load(dde)
    z_dims = list(weights()['fc21.bias'].size())[0] # Decoder input
    x_dims = list(weights()['fc4.bias'].size())[0]  # Decoder output
    print('\n[*] Using Encoding with ', str(z_dims), ' Latent Dimensions [*]')
    vae = VecVAE(x_dims,z_dims)
    vae.load_state_dict(weights())

    encoding = vae.express
    n_dofs = z_dims
    opts.set('bounds',[-5,5])

  else:
    print('[**] Using direct genotype-phenotype mapping [**]')
    encoding = enc_direct # Identify function

  # Tiny run testing
  #opts.set('maxfevals',1000)

# -- Run CMA-ES ----------------------------------------------------------- -- #
 
  for i in range(target.shape[0]):
    result = runCma(opts, n_dofs, target[i,:], evaluate, encoding)
    x = result[4]
    print('[---]')
    IK(x,evaluate,target[i],encoding,verbose=True)
    print('Genotype:\n',x)
    print('Phenotype:\n',encoding(x))
    best = encoding(x)

    print('nEvals: ' ,result[0])

    if i == 0:
      n_eval = np.c_[ result[0] ]
      f_best = np.c_[ result[1] ]
      d_dist = np.c_[ result[2] ]
      f_orig = np.c_[ result[3] ]
      x_best = np.c_[ best ]
    else:
      n_eval = np.c_[n_eval, result[0] ]
      f_best = np.c_[f_best, result[1] ]
      d_dist = np.c_[d_dist, result[2] ]
      f_orig = np.c_[f_orig, result[3] ]
      x_best = np.c_[x_best, best ]

    # n_eval = np.array(result[0])
    # f_best = np.array(result[1])
    # d_dist = np.array([result[2]])
    # f_orig = np.array([result[3]])
    # x_best = np.array(best)

    np.savetxt('n_eval_'+str(i)+'.csv',n_eval, delimiter=',')
    np.savetxt('f_best_'+str(i)+'.csv',f_best, delimiter=',')
    np.savetxt('d_dist_'+str(i)+'.csv',d_dist, delimiter=',')
    np.savetxt('f_orig_'+str(i)+'.csv',f_orig, delimiter=',')
    np.savetxt('x_best_'+str(i)+'.csv',x_best, delimiter=',')




# ------------------------------------------------------------------------- -- #
if __name__ == '__main__':
  main()



