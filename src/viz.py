import os
import numpy as np
import torch
import imageio
from pathlib import Path


def _to_uint8(frame_f32):
    return (np.clip(frame_f32, 0.0, 1.0) * 255.0).astype(np.uint8)


def rollout_real(dreamer, env, config, max_steps, noise_scale, device):
    """Roll out the current policy in the real env. Mirrors trainer.py interaction loop.

    Returns dict with frames (HWC float32 in [0,1]), actions, rewards.
    Convention: frames[t] = obs at state t (before action[t]); actions[t] = action taken at state t.
    """
    action_bound_min = torch.from_numpy(env.action_spec().minimum.copy()).to(device).float()
    action_bound_max = torch.from_numpy(env.action_spec().maximum.copy()).to(device).float()

    step = env.reset()
    h = torch.zeros(1, config.model.hidden_dim, device=device)
    z = torch.zeros(1, config.model.z_dim, device=device)
    a = torch.zeros(1, config.env.action_dim, device=device)

    frames, actions, rewards = [], [], []
    while not step.last() and len(frames) < max_steps:
        pixels = step.observation['pixels']
        frames.append(pixels.astype(np.float32))

        o = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device).float()
        h, z = dreamer.rssm(h=h, z=z, action=a, obs=dreamer.encoder(o))
        a = dreamer.policy_model(torch.cat([h, z], dim=1))
        a = (a + torch.randn_like(a) * noise_scale).clamp(action_bound_min, action_bound_max)

        total_rew = 0.0
        for _ in range(config.env.action_repeat):
            step = env.step(a.cpu().squeeze(0).numpy())
            total_rew += step.reward
            if step.last():
                break

        actions.append(a.cpu().squeeze(0).numpy())
        rewards.append(total_rew)

    return {'frames': frames, 'actions': actions, 'rewards': rewards}


def rollout_imagined(dreamer, real_frames, real_actions, warmup, imagine_steps, config, device):
    """Replay recorded (frames, actions) through the model: posterior for [0, warmup), prior for [warmup, end).

    Returns dict with decoded (list of HWC float32) and real_aligned (the first warmup+imagine_steps real frames).
    """
    total = warmup + imagine_steps
    assert len(real_frames) >= total, f"need at least {total} real frames, got {len(real_frames)}"

    h = torch.zeros(1, config.model.hidden_dim, device=device)
    z = torch.zeros(1, config.model.z_dim, device=device)
    prev_action = torch.zeros(1, config.env.action_dim, device=device)

    decoded = []
    for t in range(total):
        if t < warmup:
            o = torch.from_numpy(real_frames[t]).permute(2, 0, 1).unsqueeze(0).to(device).float()
            h, z = dreamer.rssm(h=h, z=z, action=prev_action, obs=dreamer.encoder(o))
        else:
            h, z = dreamer.rssm(h=h, z=z, action=prev_action)

        latent = torch.cat([h, z], dim=1)
        frame = dreamer.decoder(latent).squeeze(0).permute(1, 2, 0).cpu().numpy()
        decoded.append(frame.astype(np.float32))

        prev_action = torch.from_numpy(real_actions[t]).unsqueeze(0).to(device).float()

    return {
        'decoded': decoded,
        'real_aligned': [f.astype(np.float32) for f in real_frames[:total]],
        'warmup': warmup,
    }


def make_real_gif_frames(frames):
    return [_to_uint8(f) for f in frames]


def make_imagined_gif_frames(real, decoded, warmup):
    """Stacked (real on top, decoded on bottom) gif frames with a colored border on the decoded panel.

    Green border during posterior warmup, red border during pure-prior imagination.
    """
    assert len(real) == len(decoded)
    out = []
    h, w = real[0].shape[:2]
    sep = np.zeros((1, w, 3), dtype=np.uint8)
    for t in range(len(real)):
        top = _to_uint8(real[t])
        bot = _to_uint8(decoded[t]).copy()

        color = np.array([0, 200, 0], dtype=np.uint8) if t < warmup else np.array([220, 30, 30], dtype=np.uint8)
        bot[:2, :, :] = color
        bot[-2:, :, :] = color
        bot[:, :2, :] = color
        bot[:, -2:, :] = color

        frame = np.concatenate([top, sep, bot], axis=0)
        out.append(frame)
    return out


def save_gif(frames, path, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, duration=1.0 / fps, loop=0)
    return str(path)


def visualize(dreamer, env, config, warmup, imagine_steps, fps, output_dir, real_max_steps, noise_scale=0.0):
    """Run real + imagined rollout, build gif frames, write to disk, return everything needed for wandb."""
    device = next(dreamer.parameters()).device
    output_dir = Path(output_dir)

    dreamer.eval()
    with torch.no_grad():
        real = rollout_real(dreamer, env, config, max_steps=real_max_steps, noise_scale=noise_scale, device=device)
        min_needed = warmup + imagine_steps
        if len(real['frames']) < min_needed:
            raise RuntimeError(f"real rollout too short: got {len(real['frames'])} frames, need {min_needed}")

        imag = rollout_imagined(dreamer, real['frames'], real['actions'], warmup, imagine_steps, config, device)

    real_u8 = make_real_gif_frames(real['frames'])
    imag_u8 = make_imagined_gif_frames(imag['real_aligned'], imag['decoded'], warmup)

    real_path = save_gif(real_u8, output_dir / 'real.gif', fps)
    imag_path = save_gif(imag_u8, output_dir / 'imagined.gif', fps)

    real_arr = np.stack([f.astype(np.float32) / 255.0 for f in imag['real_aligned']])
    dec_arr = np.stack([f.astype(np.float32) for f in imag['decoded']])
    warmup_mse = float(((real_arr[:warmup] - np.clip(dec_arr[:warmup], 0, 1)) ** 2).mean())
    imagine_mse = float(((real_arr[warmup:] - np.clip(dec_arr[warmup:], 0, 1)) ** 2).mean())

    return {
        'real_path': real_path,
        'imagined_path': imag_path,
        'real_frames_u8': real_u8,
        'imagined_frames_u8': imag_u8,
        'episode_return': float(sum(real['rewards'])),
        'num_real_steps': len(real['frames']),
        'warmup_mse': warmup_mse,
        'imagine_mse': imagine_mse,
    }
