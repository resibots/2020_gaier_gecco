import sys
from vae import *

def main():
# -- Parse input arguments ----------------------------------------------------#
# python3 train_dde.py <archive> <n_latent> <train_rounds> <epochs> <outfile.pt>
# python3 ../../train_dde.py hex_archiveMap.npy    128 50 1000 hex_dde2.pt
# python3 ../../train_dde.py arm200_archive.npy 32 50 1000 arm200_dde.pt
# python3 ../../train_dde.py arm20_archive.npy 8 50 1000 arm20_dde.pt

  archive = sys.argv[1]
  nZ      = int(sys.argv[2])
  rounds  = int(sys.argv[3])
  epochs  = int(sys.argv[4])
  outfile = sys.argv[5]

  archive = np.load(archive)
  x = archive[:,3:]

# -- Train VAE ----------------------------------------------------------------#
  vae = VecVAE(x.shape[1], nZ)
  vae.loss_function = vae.loss_functionA
  data = vae.dataset(x)
  dataloader = DataLoader(data,batch_size=x.shape[0],shuffle=True,num_workers=1)

  loss = []
  for i in range(rounds):
      with torch.enable_grad():     
        loss.append(vae.fit(dataloader, epochs, returnLoss=True,viewMod=epochs))

# -- Save VAE ----------------------------------------------------------------#
  np.savetxt('log_train_dde.csv', loss, fmt='%.4e', delimiter='\n')
  vae.save(outfile)

if __name__ == '__main__':
  main()