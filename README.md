# VAE-Elites

Experiments with VAE Accelerated MAP-Elites and Data-Driven Encodings

---

## Dependencies

* scipy (for CVT)
* pytorch (for VAE)
* xfoil (for airfoil evaluation)

## Useful scripts
- `sh exp/nancy/sync_data.sh`


#### Notes on XFoil
To allow for truly headless evaluation a simple edit must be made to the 
Fortran code before compiling. In pyfoil/xfoil/ there are instructions on what edit has to be made if you need
to compile from source, and the extra steps needed if you need to compile from
source on Mac. First though, try to just use these binaries:


[Precompiled mac and unix binaries of 
headless XFoil are here](https://drive.google.com/open?id=1eI0EObX7O90L_x9PwPydvI7Ko3O16kzN).

The hacky way I am using xfoil right now relies on the gnu terminal commands.
On Linux machines this is not a problem, but on Mac ensure these are installed 
(`brew install coreutils`) if they are not already.

# Notes
The CVT can fail because of a big in MKL/conda (OMP error #13). In that case:
`KMP_INIT_AT_FORK=FALSE python3 ./dde_arm.py vae 10 1500 2`

See: https://github.com/ContinuumIO/anaconda-issues/issues/11294 and https://github.com/numpy/numpy/issues/11734