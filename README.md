## Benchmarking Reinforcement Learning Algorithms on Realistic Simulated Environments

Thesis by Jan Küblbeck

### Guide

#### Installation:

1. Install Mujoco and SimulationFramework as described in https://github.com/ALRhub/SimulationFramework
2. For door and soccer tasks:
   1. Add object files from Meta-World to SimulationFramework
   2. Copy xml files from envs/objects to SimulationFramework
3. For experiments: `pip install cw2`, documentation at https://github.com/ALRhub/cw2

#### Repo Structure

* `demo`: demo and testing of code
* `envs`: implementations of environments
* `envs_application`:
* `experiments`: cw2 experiments and config files for BwUniCluster
* `objects`:
* `outcomes`: results are to be saved here