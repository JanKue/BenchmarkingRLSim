## Benchmarking Reinforcement Learning Algorithms on Realistic Simulated Environments

Bachelor's thesis by Jan Küblbeck

[Link to GitHub](https://github.com/JanKue/BenchmarkingRLSim)

### Guide

#### Installation:

1. Install Mujoco and SimulationFramework as described in [SimulationFramework](https://github.com/ALRhub/SimulationFramework)
2. For door and soccer tasks:
   1. Copy necessary object files and textures from [Meta-World](https://github.com/Farama-Foundation/Metaworld) to `SimulationFramework/models/mujoco/objects`
   2. Copy xml files from `objects` folder to `SimulationFramework/models/mujoco/objects`
   3. Fill out correct path to your installation of `SimulationFramework` in the Python files in `objects`
3. For experiments: `pip install cw2`, documentation at [cw2](https://github.com/ALRhub/cw2)

#### Repo Structure

* `demo`: demo and testing of code
* `envs`: implementations of environments
* `experiments`: cw2 experiments and config files for BwUniCluster
* `objects`: code for required object assets
* `outcomes`: results are to be saved here
* `training`: code for training and testing models