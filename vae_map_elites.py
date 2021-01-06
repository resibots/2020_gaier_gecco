import numpy as np
import math
import collections
import random
import multiprocessing
import scipy.spatial
import time
import itertools


import pymap_elites.map_elites as map_elites
from vae import *

# - ------- -------- -------------------------- - #

def rec_cross_vec(xx, archive, params, vae):
  nInd = np.shape(xx)[0]
  do_cross = np.random.random(nInd)<params["prob_xover"]
  x1 = xx[do_cross,:]
  x2 = xx[~do_cross,:]

  # -- Reconstructive crossover
  rec = np.empty(np.shape(x1))
  if np.any(do_cross):
    rec = vae.getRecon(x1)
    y = (rec + x1) / 2.0
    rec = np.clip(y, a_min=params["min"][0], a_max=params["max"][0]) # assumes uniform range
    assert(rec.shape == x1.shape)

  # -- Line Mutation (# TODO: Vectorize MAP-Elites' guts #)
  mut = np.empty(np.shape(x2))
  if np.any(~do_cross):
    for i in range(np.shape(x2)[0]):
      mut[i,:] = map_elites.variation(x2[i,:], archive, params)

  children = np.r_[rec,mut]
  assert(children.shape == xx.shape)
  return children

def rec_cross(x, archive, params, vae):
  if random.random() > params["prob_xover"]:
    return map_elites.variation(x, archive, params)
  else:
    # mean of genome and reconstruction
    recons = vae.getRecon(x)
    y = (recons + x) / 2.0
    rec = np.clip(y, a_min=params["min"][0:len(y)],       \
                     a_max=params["max"][0:len(y)])
    rec = np.ravel(rec)
    assert(rec.shape == x.shape)
    return rec

# successes dict with value -> successes
def opt_bandit(results, params):
  n = len(results)
  v = list(itertools.product(params["bandit_prob_xover"], params["bandit_line_sigma"]))

  successes = collections.defaultdict(list)
  for t, s in results:
      successes[t] += [s]

  # now things should be like before
  if len(successes.keys()) < len(v): # we did not try everything yet
      unused = []
      for k in v:
          if k not in successes:
              unused += [k]
      return random.choice(unused)

  ucb = []
  for k in v:
    x = successes[k]
    mean = sum(x) / float(len(x)) * 100
    n_a = len(x)
    ucb += [mean +  math.sqrt(2 * math.log(n) / n_a)]

  a = np.argmax(ucb)
  t_size = v[a]
  return t_size

def pop_to_array(archive):
  keys = list(archive.keys())
  dim = archive[keys[0]].x.shape[0]
  data = np.full( (len(keys),dim), np.nan )
  i = 0
  for k in keys:
    data[i, :] = archive[k].x
    i += 1
  return data


def vae_map_elites(dim_map, dim_x, f, params, n_niches=1000, n_gen=1000,
                   gen=0, archive={}, centroids=np.empty(shape=(0,0)),pool=None,
                   model=VecVAE,latent_length=10, log_file=None, vae_log=None):


  # - Init --------------------------------------------------------------------#
  if pool==None:
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

  if archive == {}: # initialize map if one doesn't exist already
    archive, centroids, s = map_elites.compute(dim_map, dim_x,
                            f, n_niches=n_niches, n_gen=1,
                            log_file=log_file, centroids=centroids)

  # Bandit
  params["prob_xover"] = params["bandit_prob_xover"][0]
  params["sigma_line"] = params["bandit_line_sigma"][0]
  successes = collections.deque(maxlen=params["banditWindow"])

  # VAE
  loss = np.nan;
  vae = model(dim_x, latent_length) # get rid of this
  params["variation"]  = lambda x, archive, params: \
                          rec_cross_vec(x, archive, params, vae)

  # - Main Loop ---------------------------------------------------------------#
  for i in range(1, int(n_gen/params["trainMod"])):
    # - MAP-Elites ------------------------------------------------------------#
    with torch.no_grad():
      archive, centroids, s = map_elites.compute(dim_map, dim_x, f,
                                  n_niches=n_niches, n_gen=params["trainMod"],
                                  archive=archive, centroids=centroids,
                                  gen=i*params["trainMod"],
                                  pool=pool, log_file=log_file)
      successes += [((params["prob_xover"], params["sigma_line"]), s)]

    params["prob_xover"], params["sigma_line"] = opt_bandit(successes, params)

    # - VAE -------------------------------------------------------------------#
    v_start_time = time.time()

    # --- Convert Archive into PyTorch Dataset --------------------------------#
    dataset = vae.dataset(pop_to_array(archive))
    dataloader = DataLoader(dataset,batch_size=64,shuffle=True,num_workers=1)
    with torch.enable_grad():
      loss = vae.fit(dataloader, params["trainEpoch"],returnLoss=True)

    # --- Create VAE Crossover Operator ---------------------------------------#
    with torch.no_grad():
      vae.eval()
      params["variation"]  = lambda x, archive, params: \
                              rec_cross_vec(x, archive, params, vae)
    v_time = time.time()-v_start_time



    # - Logging and Reporting -------------------------------------------------#
    if vae_log:
      vae_log.write('{} {} {} {}\n'.format(
        params["prob_xover"], params["sigma_line"], np.nanmean(loss), v_time))
      vae_log.flush()

    if (i % params["dump_period"]) == 0:
      vae.save('vae.pt')

  # - Save final result -------------------------------------------------------#
  map_elites.__save_archive(archive, i, params['save_format'])
  vae.save('vae.pt')
  return archive, centroids, successes
