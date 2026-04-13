import os
import random
import numpy as np
import torch
from types import SimpleNamespace
from tqdm import tqdm
from src.envs import init_env

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, torch.device):
        return str(ns)
    return ns

def create_trajectory(env_name, task_name, action_dim, action_repeat, seed):
    env = init_env(env_name, task_name, seed=seed)
    action_spec = env.action_spec()
    step = env.reset()
    obs_list, act_list, rew_list, cont_list = [], [], [], []

    while not step.last():
        obs_list.append(step.observation['pixels'])
        action = np.random.uniform(action_spec.minimum, action_spec.maximum)
        total_rew = 0.0
        for _ in range(action_repeat):
            step = env.step(action)
            total_rew += step.reward
            if step.last(): break
        act_list.append(action)
        rew_list.append(total_rew)
        cont_list.append(step.discount)

    return {
        'observation': np.array(obs_list),
        'action': np.array(act_list),
        'reward': np.array(rew_list),
        'continuation': np.array(cont_list)
    }

def save_episodes(save_dir, num_episodes, env_name, task_name, action_dim, action_repeat, base_seed):
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(num_episodes), desc="generating episodes"):
        episode = create_trajectory(env_name, task_name, action_dim, action_repeat, seed=base_seed + i)
        np.savez(os.path.join(save_dir, f'seed_{i}.npz'), **episode)

def load_episodes(episode_dir):
    episodes = []
    for file in os.listdir(episode_dir):
        if not file.endswith('npz'):
            continue
        data = np.load(os.path.join(episode_dir, file))
        episodes.append({k: data[k] for k in data.files})
    return episodes

def sample_batch(episodes, batch_size, sequence_length):
    batch = []
    for _ in range(batch_size):
        episode = episodes[np.random.choice(len(episodes))]
        len_episode = episode['observation'].shape[0]
        start_idx = np.random.randint(0, len_episode - sequence_length)
        end_idx = start_idx + sequence_length
        batch.append({
            'observation': episode['observation'][start_idx:end_idx],
            'action': episode['action'][start_idx:end_idx],
            'reward': episode['reward'][start_idx:end_idx],
            'continuation': episode['continuation'][start_idx:end_idx]
        })

    return {
        'observation': torch.from_numpy(np.stack([e['observation'] for e in batch])),
        'action': torch.from_numpy(np.stack([e['action'] for e in batch])).float(),
        'reward': torch.from_numpy(np.stack([e['reward'] for e in batch])).float(),
        'continuation': torch.from_numpy(np.stack([e['continuation'] for e in batch])).float()
    }

def compute_lambda_value(rewards, values, l=0.95, discount_factor=0.99):
    batch_size, num_steps = rewards.shape[:2]
    estims = [values[:, -1]]
    for t in range(num_steps - 2, -1, -1):
        estims.insert(0, rewards[:, t] + discount_factor * ((1 - l) * values[:, t + 1] + l * estims[0]))
    return torch.stack(estims, dim=1)
