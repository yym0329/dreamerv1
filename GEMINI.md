# DreamerV1 Project Overview

A modular and configurable PyTorch implementation of the **DreamerV1** algorithm.

## Folder Structure
- `src/`: Core implementation logic.
  - `models.py`: RSSM and Actor-Critic architectures.
  - `envs.py`: Environment wrappers.
  - `utils.py`: Data handling and value estimation.
  - `trainer.py`: Training logic.
- `configs/`: YAML-based hyperparameter management.
- `experiments/`: (Auto-generated) Checkpoints and dataset.
- `train.py`: Main entry point for training.
- `test.py`: Entry point for evaluation.

## Configuration (YAML)
All hyperparameters are managed in `configs/default.yaml`. You can create multiple config files for different experiments.

## Running the Project

### Installation
```bash
pip install dm_control mujoco wandb tqdm torch numpy pyyaml
apt-get install -y libosmesa6-dev
```

### Training
```bash
# Using default config
python train.py

# Using custom config or overrides
python train.py --config configs/my_experiment.yaml --exp_dir ./exp_debug
```

### Evaluation
```bash
python test.py --checkpoint ./experiments/checkpoints/50_dreamer.pt
```

## Features
- **YAML Configuration**: Nested and clean hyperparameter management.
- **Auto-Device Detection**: Seamlessly switches between CUDA and CPU.
- **W&B Integration**: Detailed experiment tracking.
- **Headless Optimized**: Ready for `tmux` with `osmesa` rendering.
