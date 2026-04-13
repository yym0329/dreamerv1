import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import argparse
import copy
import traceback
import yaml
import torch
from pathlib import Path
from src.trainer import train_dreamer
from src.utils import dict_to_namespace

def main():
    parser = argparse.ArgumentParser(description="DreamerV1 serial seed sweep")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Base config file')
    parser.add_argument('--seeds', type=int, nargs='+', required=True, help='Seeds to run, e.g. --seeds 0 1 2')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        base_config_dict = yaml.safe_load(f)

    base_exp_dir = Path(base_config_dict['exp_dir'])

    failures = []
    for seed in args.seeds:
        config_dict = copy.deepcopy(base_config_dict)
        config_dict['seed'] = seed
        config_dict['exp_dir'] = str(base_exp_dir / f'seed_{seed}')
        config = dict_to_namespace(config_dict)

        if config.device == "auto":
            config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            config.device = torch.device(config.device)

        print(f"\n{'=' * 60}")
        print(f"[sweep] seed={seed}  exp_dir={config.exp_dir}  device={config.device}")
        print(f"{'=' * 60}")

        try:
            train_dreamer(config)
        except Exception:
            traceback.print_exc()
            failures.append(seed)
            print(f"[sweep] seed={seed} failed; continuing with next seed")

    print(f"\n[sweep] done. ran {len(args.seeds)} seeds, {len(failures)} failed: {failures}")

if __name__ == "__main__":
    main()
