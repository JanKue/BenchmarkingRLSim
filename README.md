## Benchmarking Reinforcement Learning Algorithms on Realistic Simulated Environments

Bachelor's thesis by Jan Küblbeck

### Guide

#### Installation:

1. Install Mujoco and SimulationFramework as described in [SimulationFramework](https://github.com/ALRhub/SimulationFramework)
2. For door and soccer tasks: Manually copy required meshes and textures from [Meta-World](https://github.com/Farama-Foundation/Metaworld) to `SimulationFramework/models/mujoco/objects`
3. For experiments: `pip install cw2`, documentation at [cw2](https://github.com/ALRhub/cw2)

#### Repo Structure

* `custom_envs`: implementations of environments
* `demo`: demo and testing of code
* `experiments`: cw2 experiments and config files for BwUniCluster
* `objects`: code for required object assets
* `outcomes`: results are to be saved here
* `pdfs`: thesis and related PDFs (see here for additional documentation)
* `training`: code for training and testing models

[This repo on GitHub](https://github.com/JanKue/BenchmarkingRLSim)