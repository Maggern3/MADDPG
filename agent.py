from networks import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy

class Agent():
    def __init__(self, actor_size, action_size, critic_size):
        super().__init__() 
        gpu = torch.cuda.is_available()
        if(gpu):
            print('GPU/CUDA works! Happy fast training :)')
            torch.cuda.current_device()
            torch.cuda.empty_cache()
            self.device = torch.device("cuda")
        else:
            print('training on cpu...')
        self.device = torch.device("cpu")

        self.actor = Actor(actor_size, action_size).to(self.device)
        self.actor_target = Actor(actor_size, action_size).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic = Critic(critic_size).to(self.device)
        self.critic_target = Critic(critic_size).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.001, weight_decay=0)        
        self.gamma = 0.95#0.99
        self.tau = 0.001        
        self.noise = OUNoise((action_size), 2)
        self.target_network_update(self.actor_target, self.actor, 1.0)
        self.target_network_update(self.critic_target, self.critic, 1.0)

    def select_actions(self, state):
        state = torch.from_numpy(state).float().to(self.device).view(1, -1)
        #print(state.shape)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state).cpu().data.squeeze(0)
        self.actor.train()
        actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def target_network_update(self, target_network, network, tau):
        for network_param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(tau * network_param.data + (1.0-tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.05):#0.1 0.05
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(torch.rand(x.shape))
        self.state = x + dx
        return torch.tensor(self.state).float()