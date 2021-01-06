# Pycontrollers

This repository stores python robot controllers. It is used in several other packages such as:

* [ros_pyminitaur](https://github.com/resibots/ros_pyminitaur.git) - The ros package with everything needed to read maps created with Map Elites and send the commands to the minitaur through ros

* [pybullet_minitaur_sim](https://github.com/resibots/pybullet_minitaur_sim) - This contains a pybullet minitaur simulation based on [this](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/minitaur/envs) github repository . The pybullet_minitaur_sim defines everything needed to run episodes and create maps with Map Elites.

* [pyite](https://github.com/resibots/pyite) - An IT&E implementation in python, it can be used with pybullet for simulation or with ros to send commands to the real robots. It is selecting the best behaviors in the maps created with Map Elites thanks to a bayesian optimization process
