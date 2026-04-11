# DreamerV1 PyTorch Implementation

A modular and highly configurable PyTorch implementation of the **DreamerV1** algorithm ("Dream to Control: Learning Behaviors by Latent Imagination").

## Project Structure

The codebase is organized into a clean, modular structure:

```text
.
├── train.py                # Main entry point for training
├── test.py                 # Entry point for evaluation
├── configs/
│   └── default.yaml        # Default hyperparameters and settings
├── src/                    # Core source code
│   ├── models.py           # Neural network architectures (RSSM, Encoder, etc.)
│   ├── envs.py             # Environment wrappers and initialization
│   ├── trainer.py          # Core training loop logic
│   └── utils.py            # Data handling and value estimation
├── experiments/            # Auto-generated: stores checkpoints and datasets
└── GEMINI.md               # Quick context for AI agents
```

## Features

- **Modular Design**: Separate modules for models, environments, and training logic.
- **YAML Configuration**: Easy hyperparameter management via nested YAML files.
- **Headless Optimized**: Default support for OSMesa rendering, making it ready for `tmux` and remote servers.
- **W&B Integration**: Full experiment tracking with Weights & Biases.
- **Auto-Device Detection**: Seamlessly switches between CUDA and CPU.

## Installation

### 1. Dependencies
Ensure you have Python 3.12+ installed. Install the required Python packages:

```bash
pip install dm_control mujoco wandb tqdm torch numpy pyyaml
```

### 2. System Packages (for Headless Rendering)
If you are running on a server without a display (e.g., via SSH or in a Docker container), install OSMesa:

```bash
apt-get update -qq && apt-get install -y libosmesa6-dev
```

## Usage

### Training

To start training with the default configuration:
```bash
python train.py
```

To use a custom configuration file or override the experiment directory:
```bash
python train.py --config configs/my_config.yaml --exp_dir ./my_experiment
```

#### Key Training Arguments:
- `--config`: Path to the YAML configuration file (default: `configs/default.yaml`).
- `--exp_dir`: Directory to save logs, checkpoints, and datasets.
- `--resume`: If provided, the script will attempt to resume from the latest checkpoint in the experiment directory.

### Evaluation

To evaluate a trained model checkpoint:
```bash
python test.py --checkpoint ./experiments/checkpoints/50_dreamer.pt
```

### Configuration

All hyperparameters are managed in `configs/default.yaml`. Key sections include:
- `env`: Environment name, task, and rendering resolution.
- `model`: Dimensions for hidden states, latents, and network depth.
- `train`: Learning rates, batch sizes, sequence lengths, and imagination horizon.

## Implementation Details

- **RSSM**: Recurrent State Space Model with GRU and MLP components for prior/posterior distributions.
- **Actor-Critic**: Implemented in latent space using imagined rollouts.
- **Value Estimation**: Uses $\lambda$-return for stable value targets.
- **Observation**: Pixel-based observations (default 64x64) with normalization.

## Logging

This project uses **Weights & Biases (W&B)** for logging. On your first run, you will be prompted to log in. You can monitor:
- Loss curves (Observation, Reward, KLD, Actor, Value).
- Episode returns.
- Model reconstructions and imagined trajectories (via the wandb dashboard).
