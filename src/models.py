import torch
from torch import nn
import torch.nn.functional as F

def kl_divergence(mu_1, sigma_1, mu_2, sigma_2):
    # shape: (B, T, D)
    return (0.5*(2*(sigma_2.log() - sigma_1.log())
    + (sigma_1.pow(2)
    + (mu_1 - mu_2).pow(2))/sigma_2.pow(2) - 1
     ).sum(dim=-1))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        self.act = nn.ReLU()

    def forward(self, obs):
        x = self.act(self.conv1(obs))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        return x.reshape(x.size(0), -1)

class Decoder(nn.Module):
    def __init__(self, state_dim: int = 230):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.linear1(x)).view(x.size(0), 1024, 1, 1)
        x = self.act(self.deconv1(x))
        x = self.act(self.deconv2(x))
        x = self.act(self.deconv3(x))
        return self.deconv4(x)

class ActionModel(nn.Module):
    def __init__(self, num_layers=3, hidden_dim=300, latent_dim=230, action_dim=6):
        super().__init__()
        layers = [nn.Linear(latent_dim, hidden_dim), nn.ELU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ELU()])
        layers.append(nn.Linear(hidden_dim, 2 * action_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        raw_statistics = self.layers(x)
        mean, raw_std = torch.chunk(raw_statistics, 2, dim=-1)
        mean = 5 * torch.tanh(mean / 5)
        std = F.softplus(raw_std) + 1e-4
        eps = torch.randn_like(mean)
        return torch.tanh(mean + std * eps)

class ValueModel(nn.Module):
    def __init__(self, num_layers=3, hidden_dim=300, latent_dim=230):
        super().__init__()
        layers = [nn.Linear(latent_dim, hidden_dim), nn.ELU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ELU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class RSSM(nn.Module):
    def __init__(self, hidden_dim=200, z_dim=30, action_dim=6):
        super().__init__()
        self.gru = nn.GRUCell(input_size=z_dim + action_dim, hidden_size=hidden_dim)
        self.posterior_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1024, 300), nn.ELU(),
            nn.Linear(300, 300), nn.ELU(),
            nn.Linear(300, 2 * z_dim)
        )
        self.prior_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 300), nn.ELU(),
            nn.Linear(300, 300), nn.ELU(),
            nn.Linear(300, 2 * z_dim)
        )
        self.softplus = nn.Softplus()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

    def forward(self, h, z, action, obs=None, return_dist=False, return_both=False):
        h = self.gru(torch.cat([z, action], dim=1), h)
        if obs is None:
            mean, raw_std = torch.chunk(self.prior_mlp(h), 2, dim=-1)
        else:
            mean, raw_std = torch.chunk(self.posterior_mlp(torch.cat([h, obs], dim=1)), 2, dim=-1)

        std = self.softplus(raw_std) + 1e-4
        eps = torch.randn_like(mean)
        z = mean + std * eps

        if return_both:
            prior_mean, prior_raw_std = torch.chunk(self.prior_mlp(h), 2, dim=-1)
            prior_std = self.softplus(prior_raw_std) + 1e-4
            prior_z = prior_mean + prior_std * torch.randn_like(prior_mean)
            return h, z, mean, std, prior_z, prior_mean, prior_std

        if return_dist:
            return h, z, mean, std
        return h, z

class RewardModel(nn.Module):
    def __init__(self, state_dim=230):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(state_dim, 300), nn.ELU(),
            nn.Linear(300, 300), nn.ELU(),
            nn.Linear(300, 1)
        )

    def forward(self, s):
        return self.ffn(s)

class Dreamer(nn.Module):
    def __init__(self, hidden_dim=200, z_dim=30, action_dim=6, num_ffn_layers=3, discount_factor=0.99, free_nats:float = 3.):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(state_dim=hidden_dim + z_dim)
        self.rssm = RSSM(hidden_dim, z_dim, action_dim)
        self.reward_model = RewardModel(hidden_dim + z_dim)
        self.policy_model = ActionModel(num_layers=num_ffn_layers, hidden_dim=hidden_dim, latent_dim=hidden_dim + z_dim, action_dim=action_dim)
        self.value_model = ValueModel(num_layers=num_ffn_layers, hidden_dim=hidden_dim, latent_dim=hidden_dim + z_dim)
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.free_nats = free_nats

    def get_dynamics_model_parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.rssm.parameters()) + list(self.reward_model.parameters())

    def get_action_model_parameters(self):
        return list(self.policy_model.parameters())

    def get_value_model_parameters(self):
        return list(self.value_model.parameters())

    def compute_dynamics_loss(self, actions, observations, rewards, mode='reconstruction', beta=1.0):
        num_steps = actions.shape[1]
        batch_size = actions.shape[0]
        device = actions.device

        post_h = torch.zeros(batch_size, self.hidden_dim).to(device)
        post_z = torch.zeros(batch_size, self.z_dim).to(device)
        first_action = torch.zeros(batch_size, self.action_dim).to(device)

        states = []
        prior_dists = []
        posterior_dists = []

        obs_encodes = self.encoder(observations.reshape(-1, *observations.shape[-3:]))
        obs_encodes = obs_encodes.reshape(batch_size, num_steps, -1)

        (post_h, post_z, post_mean, post_sigma, 
         prior_z, prior_mean, prior_sigma) = self.rssm(h=post_h, z=post_z, action=first_action, obs=obs_encodes[:, 0, :], return_dist=True, return_both=True)
        
        states.append((post_h, post_z, prior_z))
        prior_dists.append((prior_mean, prior_sigma))
        posterior_dists.append((post_mean, post_sigma))

        for t in range(1, num_steps):
            (post_h, post_z, post_mean, post_sigma, 
             prior_z, prior_mean, prior_sigma) = self.rssm(h=post_h, z=post_z, action=actions[:, t-1, :], obs=obs_encodes[:, t, :], return_dist=True, return_both=True)
            states.append((post_h, post_z, prior_z))
            prior_dists.append((prior_mean, prior_sigma))
            posterior_dists.append((post_mean, post_sigma))

        hts = torch.stack([s[0] for s in states], dim=1)
        zts = torch.stack([s[1] for s in states], dim=1)
        latents = torch.cat([hts, zts], dim=-1)
        
        obs_decodeds = self.decoder(latents.reshape(-1, latents.size(-1))).reshape(batch_size, num_steps, *observations.shape[-3:])
        reward_preds = self.reward_model(latents.reshape(-1, latents.size(-1))).reshape(batch_size, num_steps, -1)

        post_means = torch.stack([m for m, _ in posterior_dists], dim=1)
        post_stds = torch.stack([s for _, s in posterior_dists], dim=1)
        prior_means = torch.stack([m for m, _ in prior_dists], dim=1)
        prior_stds = torch.stack([s for _, s in prior_dists], dim=1)

        observation_loss = F.mse_loss(observations, obs_decodeds)
        reward_loss = F.mse_loss(rewards.unsqueeze(-1), reward_preds)
        kld_elements = kl_divergence(post_means, post_stds, prior_means, prior_stds)
        dynamics_loss = torch.maximum(kld_elements, torch.tensor(self.free_nats, device=device)).mean()
        loss = observation_loss + reward_loss + beta * dynamics_loss

        metrics = {'Loss/Observation': observation_loss.item(), 'Loss/Reward': reward_loss.item(), 'Loss/KLD': kld_elements.mean().item(), 'Loss/Total': loss.item()}
        return loss, hts, zts, metrics
