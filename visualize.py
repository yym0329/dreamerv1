import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import argparse
import re
import yaml
import torch
from pathlib import Path
from src.models import Dreamer
from src.envs import init_env
from src.utils import dict_to_namespace, set_seed
from src.viz import visualize


def _infer_config_path(checkpoint_path):
    return Path(checkpoint_path).parent.parent / 'config.yaml'


def _infer_step(checkpoint_path):
    m = re.match(r'(\d+)_dreamer', Path(checkpoint_path).stem)
    return m.group(1) if m else 'unknown'


def main():
    parser = argparse.ArgumentParser(description="DreamerV1 visualization: real + imagined rollouts")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to *_dreamer.pt')
    parser.add_argument('--config', type=str, default=None, help='Config yaml (default: <exp_dir>/config.yaml)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output dir (default: <ckpt parent>/viz_<step>/)')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--imagine-steps', type=int, default=45)
    parser.add_argument('--real-max-steps', type=int, default=200)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    config_path = Path(args.config) if args.config else _infer_config_path(checkpoint_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            f"Pass --config explicitly, or place the training config next to the experiment dir."
        )

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)

    device_str = getattr(config, 'device', 'auto')
    if device_str == 'auto' or not isinstance(device_str, str):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    config.device = device

    set_seed(args.seed)

    env = init_env(config.env.name, config.env.task, seed=args.seed)

    dreamer = Dreamer(
        hidden_dim=config.model.hidden_dim,
        z_dim=config.model.z_dim,
        action_dim=config.env.action_dim,
        num_ffn_layers=config.model.num_ffn_layers,
        discount_factor=config.model.discount_factor,
        free_nats=config.train.free_nats,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    dreamer.load_state_dict(ckpt['model_state_dict'])
    dreamer.eval()
    print(f"[viz] loaded {checkpoint_path} on {device}")

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / f'viz_{_infer_step(checkpoint_path)}'

    result = visualize(
        dreamer, env, config,
        warmup=args.warmup,
        imagine_steps=args.imagine_steps,
        fps=args.fps,
        output_dir=output_dir,
        real_max_steps=args.real_max_steps,
        noise_scale=0.0,
    )

    print(f"[viz] real gif:     {result['real_path']}")
    print(f"[viz] imagined gif: {result['imagined_path']}")
    print(f"[viz] real steps:     {result['num_real_steps']}")
    print(f"[viz] episode return: {result['episode_return']:.2f}")
    print(f"[viz] warmup MSE:     {result['warmup_mse']:.5f}")
    print(f"[viz] imagine MSE:    {result['imagine_mse']:.5f}")


if __name__ == "__main__":
    main()
