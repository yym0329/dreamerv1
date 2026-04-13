import os
import torch
import numpy as np
import wandb
from tqdm import tqdm
from pathlib import Path
from src.models import Dreamer
from src.utils import save_episodes, load_episodes, sample_batch, compute_lambda_value
from src.envs import init_env

def train_dreamer(config):
    exp_dir = Path(config.exp_dir)
    ckpt_dir = exp_dir / 'checkpoints'
    dataset_dir = exp_dir / 'dataset'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not any(dataset_dir.iterdir()):
        save_episodes(dataset_dir, config.train.num_seed_episodes, config.env.name, config.env.task, config.env.action_dim, config.env.action_repeat)
    
    episodes = load_episodes(dataset_dir)
    env = init_env(config.env.name, config.env.task)
    action_bound_min = torch.from_numpy(env.action_spec().minimum.copy()).to(config.device).float()
    action_bound_max = torch.from_numpy(env.action_spec().maximum.copy()).to(config.device).float()

    dreamer = Dreamer(
        hidden_dim=config.model.hidden_dim, 
        z_dim=config.model.z_dim, 
        action_dim=config.env.action_dim, 
        num_ffn_layers=config.model.num_ffn_layers, 
        discount_factor=config.model.discount_factor,
        free_nats= config.train.free_nats
    ).to(config.device)

    dyn_opt = torch.optim.Adam(dreamer.get_dynamics_model_parameters(), lr=config.train.dynamics_model_lr)
    act_opt = torch.optim.Adam(dreamer.get_action_model_parameters(), lr=config.train.action_model_lr)
    val_opt = torch.optim.Adam(dreamer.get_value_model_parameters(), lr=config.train.value_model_lr)

    start_step = 0
    wandb_run_id = None
    if config.resume:
        ckpts = list(ckpt_dir.glob('*.pt'))
        if ckpts:
            latest_ckpt = max(ckpts, key=lambda x: int(x.stem.split('_')[0]))
            checkpoint = torch.load(latest_ckpt)
            dreamer.load_state_dict(checkpoint['model_state_dict'])
            dyn_opt.load_state_dict(checkpoint['dyn_opt'])
            act_opt.load_state_dict(checkpoint['act_opt'])
            val_opt.load_state_dict(checkpoint['val_opt'])
            start_step = checkpoint['num_interactions'] + 1
            wandb_run_id = checkpoint.get('wandb_run_id')

    wandb.init(project="dreamerV1", config=vars(config), id=wandb_run_id, resume="allow")
    wandb.watch(dreamer, log='all', log_freq=100)
    for step_idx in range(start_step, config.train.max_steps):
        # Training Phase
        dreamer.train()
        for _ in tqdm(range(config.train.collect_interval), desc=f"Step {step_idx}: Training", leave=False):
            batch = sample_batch(episodes, config.train.batch_size, config.train.seq_length)
            obs = batch['observation'].to(config.device).permute(0, 1, 4, 2, 3)
            actions = batch['action'].to(config.device)
            rewards = batch['reward'].to(config.device)

            # Dynamics Update
            loss, hts, zts, metrics = dreamer.compute_dynamics_loss(actions, obs, rewards, beta=config.train.beta)
            dyn_opt.zero_grad()
            loss.backward()
            dyn_opt.step()

            # Actor-Value Update
            for p in dreamer.get_dynamics_model_parameters(): p.requires_grad_(False)
            
            hts_ = hts.detach().reshape(-1, config.model.hidden_dim)
            zts_ = zts.detach().reshape(-1, config.model.z_dim)
            
            rollout_h, rollout_z = [hts_], [zts_]
            for _ in range(config.train.imagination_horizon):
                ats = dreamer.policy_model(torch.cat([hts_, zts_], dim=1))
                hts_, zts_ = dreamer.rssm(h=hts_, z=zts_, action=ats)
                rollout_h.append(hts_)
                rollout_z.append(zts_)
            
            all_hts = torch.stack(rollout_h, dim=1)
            all_zts = torch.stack(rollout_z, dim=1)
            latents = torch.cat([all_hts, all_zts], dim=-1)
            
            rewards_imag = dreamer.reward_model(latents).reshape(-1, config.train.imagination_horizon + 1, 1)
            values_imag = dreamer.value_model(latents).reshape(-1, config.train.imagination_horizon + 1, 1)
            
            lambda_vals = compute_lambda_value(rewards_imag[:, 1:], values_imag[:, 1:].detach(), l=config.train.lambda_)
            
            act_loss = -torch.mean(lambda_vals)
            val_loss = 0.5 * torch.mean((lambda_vals.detach() - values_imag[:, 1:])**2)
            
            act_opt.zero_grad()
            val_opt.zero_grad()
            
            act_loss.backward(retain_graph=True)
            val_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(dreamer.get_action_model_parameters(), 100)
            torch.nn.utils.clip_grad_norm_(dreamer.get_value_model_parameters(), 100)
            
            act_opt.step()
            val_opt.step()
            
            for p in dreamer.get_dynamics_model_parameters(): p.requires_grad_(True)
            
            wandb.log({**metrics, 'Loss/Actor': act_loss.item(), 'Loss/Value': val_loss.item()})

        # Interaction Phase
        dreamer.eval()
        with torch.no_grad():
            step = env.reset()
            h = torch.zeros(1, config.model.hidden_dim).to(config.device)
            z = torch.zeros(1, config.model.z_dim).to(config.device)
            a = torch.zeros(1, config.env.action_dim).to(config.device)
            
            obs_list, act_list, rew_list, cont_list = [], [], [], []
            while not step.last():
                o = torch.from_numpy(step.observation['pixels']).permute(2,0,1).unsqueeze(0).to(config.device).float()
                obs_list.append(step.observation['pixels'])
                
                h, z = dreamer.rssm(h=h, z=z, action=a, obs=dreamer.encoder(o))
                a = dreamer.policy_model(torch.cat([h, z], dim=1))
                a = (a + torch.randn_like(a) * config.train.exploration_noise_scale).clamp(action_bound_min, action_bound_max)
                
                total_rew = 0
                for _ in range(config.env.action_repeat):
                    step = env.step(a.cpu().squeeze(0).numpy())
                    total_rew += step.reward
                    if step.last(): break
                
                act_list.append(a.cpu().squeeze(0).numpy())
                rew_list.append(total_rew)
                cont_list.append(step.discount)

            ep_rew = sum(rew_list)
            wandb.log({'Episode Return': ep_rew})
            new_ep = {'observation': np.array(obs_list), 'action': np.array(act_list), 'reward': np.array(rew_list), 'continuation': np.array(cont_list)}
            episodes.append(new_ep)
            np.savez(dataset_dir / f'ep_{len(episodes)}.npz', **new_ep)

        # Checkpoint
        if step_idx % config.train.checkpoint_save_interval == 0:
            torch.save({
                'model_state_dict': dreamer.state_dict(),
                'dyn_opt': dyn_opt.state_dict(),
                'act_opt': act_opt.state_dict(),
                'val_opt': val_opt.state_dict(),
                'num_interactions': step_idx,
                'wandb_run_id': wandb.run.id
            }, ckpt_dir / f'{step_idx}_dreamer.pt')
