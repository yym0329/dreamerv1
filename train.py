import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import argparse
import yaml
import torch
from src.trainer import train_dreamer
from src.utils import dict_to_namespace

def main():
    parser = argparse.ArgumentParser(description="DreamerV1 Training Script")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--exp_dir', type=str, help='Override experiment directory')
    parser.add_argument('--resume', action='store_true', help='Override resume flag')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = dict_to_namespace(config_dict)
    
    # Overrides
    if args.exp_dir:
        config.exp_dir = args.exp_dir
    if args.resume:
        config.resume = True
        
    if config.device == "auto":
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        config.device = torch.device(config.device)
    
    print(f"Starting DreamerV1 training on {config.device}")
    print(f"Config: {args.config}")
    
    train_dreamer(config)

if __name__ == "__main__":
    main()
