import os
os.environ['MUJOCO_GL'] = 'egl'

import argparse
import torch
import numpy as np
from pathlib import Path
from src.models import Dreamer
from src.envs import init_env

def evaluate(args):
    env = init_env(args.env_name, args.task_name)
    dreamer = Dreamer(
        hidden_dim=args.hidden_dim, 
        z_dim=args.z_dim, 
        action_dim=args.action_dim
    ).to(args.device)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        dreamer.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")

    dreamer.eval()
    with torch.no_grad():
        step = env.reset()
        h = torch.zeros(1, args.hidden_dim).to(args.device)
        z = torch.zeros(1, args.z_dim).to(args.device)
        a = torch.zeros(1, args.action_dim).to(args.device)
        
        total_reward = 0
        while not step.last():
            o = torch.from_numpy(step.observation['pixels']).permute(2,0,1).unsqueeze(0).to(args.device).float() / 255.0
            h, z = dreamer.rssm(h=h, z=z, action=a, obs=dreamer.encoder(o))
            a = dreamer.policy_model(torch.cat([h, z], dim=1))
            
            step = env.step(a.cpu().squeeze(0).numpy())
            total_reward += step.reward
            
        print(f"Evaluation Reward: {total_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--env_name', type=str, default='walker')
    parser.add_argument('--task_name', type=str, default='walk')
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--z_dim', type=int, default=30)
    parser.add_argument('--action_dim', type=int, default=6)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate(args)
