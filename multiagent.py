from agent import Agent
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class MultiAgent():
    def __init__(self, state_size, action_size, agents):
        super().__init__()
        self.networks = [Agent(state_size, action_size, state_size + action_size * agents),
                         Agent(state_size, action_size, state_size + action_size * agents)]
        self.agents = agents
        self.replay_buffer = deque(maxlen=1000000)#1m
        self.batch_size = 128 # 256
        self.device = torch.device("cpu")

    def act(self, states_all_agents):
        actions = [agent.select_actions(states) for agent, states in zip(self.networks, states_all_agents)]
        return torch.stack(actions)

    def act_train(self, states_all_agents, agnt_num):
        actions = []
        for agnt in range(self.agents):
            agent = self.networks[agnt]
            states = states_all_agents[:,agnt]
            actions_agent = agent.actor(states)
            if(agnt != agnt_num):
                actions_agent = actions_agent.detach()
            actions.append(actions_agent)
        return actions

    def act_target(self, states_all_agents):
        actions = []
        for agnt in range(self.agents):
            agent = self.networks[agnt]
            states = states_all_agents[:,agnt]
            #print(states.shape)
            agent.actor.eval()
            with torch.no_grad():
                actions_agent = agent.actor_target(states).cpu().data.numpy()
            agent.actor.train()
            actions.append(actions_agent)
        return torch.tensor(actions)

    def add(self, sars):
        self.replay_buffer.append(sars)
    
    def train(self, agnt):    
        agent = self.networks[agnt]
        if(len(self.replay_buffer) > self.batch_size): 
            states, actions, rewards, next_states, dones = self.sample(agnt)     
            next_actions = self.act_target(next_states)
            next_actions = torch.cat((next_actions[0], next_actions[1]), dim=1)
            with torch.no_grad():
                next_state_q_v = agent.critic_target(torch.cat((next_states[:,agnt], next_actions), dim=1)) # torch.Size([128, 1])
            q_targets = rewards + (agent.gamma * next_state_q_v * (1-dones))
            actions = torch.cat((actions[:,0], actions[:,1]), dim=1)
            current_q_v = agent.critic(torch.cat((states[:,agnt], actions), dim=1))

            criterion = torch.nn.SmoothL1Loss()
            critic_loss = criterion(current_q_v, q_targets.detach()) # SmoothL1Loss, q_targets.detach()
            agent.critic_optim.zero_grad()
            critic_loss.backward()
            #torch.nn.utils.clip_grad_norm(agent.critic.parameters(), 1) #0.5
            agent.critic_optim.step()

            predicted_actions = self.act_train(states, agnt)
            predicted_actions = torch.cat((predicted_actions[0], predicted_actions[1]), dim=1)
            actor_loss = -agent.critic(torch.cat((states[:,agnt], predicted_actions), dim=1)).mean()
            agent.actor_optim.zero_grad()
            actor_loss.backward()
            agent.actor_optim.step()
            agent.target_network_update(agent.actor_target, agent.actor, agent.tau)
            agent.target_network_update(agent.critic_target, agent.critic, agent.tau)

    
    def sample(self, agnt):        
        samples = random.sample(self.replay_buffer, k=self.batch_size)     
        states = torch.tensor([s[0] for s in samples]).float().to(self.device)        
        actions = torch.tensor([s[1] for s in samples]).float().to(self.device).squeeze(2)
        rewards = torch.tensor([s[2][agnt] for s in samples]).float().unsqueeze(1).to(self.device)
        next_states = torch.tensor([s[3] for s in samples]).float().to(self.device)
        dones = torch.tensor([s[4][agnt] for s in samples]).float().unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones
