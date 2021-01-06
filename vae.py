import sys
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math



device = 'cpu'
#torch.device("cuda" if args.cuda else "cpu")

# https://github.com/pytorch/examples/blob/master/vae/main.py
class VecVAE(nn.Module):
  def __init__(self, input_space, latent_space, dataset=None):
    super(VecVAE, self).__init__()
    if dataset == None:
      self.dataset = VecDataSet # Dataset of 1d vectors
    else:
      self.dataset = dataset

    self.input_space = input_space
    self.latent_space = latent_space
    self.hidden_size = latent_space * 2


    self.fc1  = nn.Linear(self.input_space, self.hidden_size)
    self.fc21 = nn.Linear(self.hidden_size, latent_space)
    self.fc22 = nn.Linear(self.hidden_size, latent_space)
    self.fc3  = nn.Linear(latent_space    , self.hidden_size)
    self.fc4  = nn.Linear(self.hidden_size, self.input_space)

    self.optimizer =  optim.Adam(self.parameters(), lr=1e-3)

    self.loss_function = self.loss_functionC

  def encode(self, x):
    h1 = F.relu(self.fc1(x))
    return self.fc21(h1), self.fc22(h1)

  def decode(self, z):
    h3 = F.relu(self.fc3(z))
    return torch.tanh(self.fc4(h3))



  # - Below are generic for any VAE: make part of a super class ---------------#

  def express(self,z):
    # Use VAE as a generator, given numpy latent vec return full numpy phenotype
    latent = torch.from_numpy(z).float()
    pheno = self.decode(latent)
    return pheno.detach().numpy()

  def getRecon(self, x):
     # Run through encoder and decoder
    l = self.encode(torch.from_numpy(x).float())[0]
    r = np.reshape(self.decode(l).detach().numpy(), x.shape)
    recon = r
    return recon

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar


  # - LOSS FUNCTIONS ------------- - #

  # | Normalize by dimensionality: NO;  KL-Annealing: YES
  def loss_functionA(self, recon_x, x, mu, logvar, input_space, fit, kl_weight=1.0):
    BCE = F.mse_loss(recon_x, x,reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
    return BCE + (kl_weight * KLD)

  # | Normalize by dimensionality: YES;  Annealing: NO
  def loss_functionB(self, recon_x, x, mu, logvar, input_space, fit, kl_weight=1.0):
    BCE = F.mse_loss(recon_x, x,reduction='sum') / (input_space)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
    return BCE + (0.1 * KLD)

  # | Normalize by dimensionality: NO;  Annealing: NO
  def loss_functionC(self, recon_x, x, mu, logvar, input_space, fit, kl_weight=1.0):
    BCE = F.mse_loss(recon_x, x,reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
    return BCE + (0.1 * KLD)

  

  # - ------------- ------------- - #

  def fit(self, dataloader, n_epoch, viewMod=100, returnLoss=False):
    loss = np.full(n_epoch+1, np.nan)
    self.train()
    for e in range(1, n_epoch+1):
      mean,std = self.epoch(e, dataloader, e/(n_epoch+1))
      loss[e] = mean
      if ((e) % viewMod) == 0:
        print('Loss at Epoch ', e, ':\t', mean)
    if returnLoss:
      return loss

  def epoch(self, epoch_id, train_loader, percDone):
    self.train()
    train_loss = []
    kl_weight = np.min([percDone*4.0,1.0])
    #print(kl_weight)
    for batch_idx, (data, fit) in enumerate(train_loader):
      data = data.to(device)
      self.optimizer.zero_grad()
      recon_batch, mu, logvar = self.forward(data)
      loss = self.loss_function(recon_batch, data, mu, logvar, self.input_space, fit, kl_weight)
      loss.backward()
      r = np.linalg.norm(recon_batch.detach().numpy() - data.detach().numpy(), axis=0)
      train_loss += [r]
      self.optimizer.step()
    per_input_loss = np.vstack(train_loss)
    return np.mean(per_input_loss), np.std(per_input_loss)    

  def save(self, path):
    torch.save(self.state_dict, path)


class VecDataSet(Dataset):
  def __init__(self, pop):
    self.data = torch.tensor(pop).float()

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    return self.data[idx,:], 0


class GaitDataSet(Dataset):
  # TODO: Test that this works!!!
  def __init__(self, fname, dofs = 12, num_actions=60, desc_size=360, min_max = 1./100.):
    self.min_max = min_max
    self.width = num_actions# 300 for states
    self.height = dofs
    archive = np.load(fname)
    # for actions (positions):
    self.data = torch.tensor(archive[:, desc_size+1:archive.shape[1]]).float()# * 0.5 + 1.0
    print(self.data.min())
    print(self.data.max())
    save_image(self.data.view(self.data.shape[0], 1, self.width, self.height), 'results/data.png', nrow=100)

    print("loaded:", self.data.shape)
    
  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    return self.data[idx, :].view(1, self.input_space), 0

#------------------------------------------------------------------------------#
if __name__ == "__main__":


  parser = argparse.ArgumentParser(description='VAE Example')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 128)')
  parser.add_argument('--epochs', type=int, default=10, metavar='N',
            help='number of epochs to train (default: 10)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
            help='enables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
  parser.add_argument('--loginterval', type=int, default=1, metavar='N',
            help='how many batches to wait before logging training status')
  parser.add_argument('--data', type=str, default="archive_40.npy", metavar='N',
        help='data')

  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  torch.manual_seed(args.seed)

  nDim = 40
  nLat = 10
  model = VecVAE(nDim, nLat)

  # Load data set
  archive = np.load(args.data)
  pop = archive[:,-nDim:]
  dataset = model.dataLoader(pop)
  dataloader = DataLoader(dataset,batch_size=args.batch_size,\
                          shuffle=True,num_workers=1)

  # Train Model  
  model.train()
  for e in range(1, args.epochs + 1):
    mean,std = model.epoch(e, dataloader)
    if (e % args.loginterval) == 0:
        print('Loss at Epoch ', e, ':\t', mean)

  a = np.random.randn(40)
  print(model.getRecon(a))