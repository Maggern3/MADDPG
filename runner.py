from unityagents import UnityEnvironment
import torch
from collections import deque
from multiagent import MultiAgent
import numpy as np
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name='Tennis_Windows_x86_64/Tennis.exe')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
observation_state_size = brain.vector_observation_space_size
action_space_size = brain.vector_action_space_size
print('states ', observation_state_size)
print('actions ', action_space_size)
state_size = 24
agents = 2

training_interval = 4
train_steps = 2 #10
agent = MultiAgent(state_size, action_space_size, agents)
scores = []
scores_last_hundred_episodes = deque(maxlen=100)
actions = np.random.randn(2, 2)
actions = np.clip(actions, -1, 1)  
#print(actions)
for episode in range(5000): # 1000-30k
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    states = env_info.vector_observations #states = env_info.vector_observations[0] 
    #print(states.shape)
    #agent.reset()
    rewards = np.zeros(agents)
    timesteps = 0
    for timestep in range(10000):
        timesteps += 1
        actions = agent.act(states).numpy()    
        #print('rnr.py actions ',actions)  
        env_info = env.step(actions)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations  #next_state = env_info.vector_observations[0]
        #print(next_state)
        reward = env_info.rewards                  # same
        done = env_info.local_done                 # same
        sars = (states, actions, reward, next_state, done)
        agent.add(sars)
        for agnt in range(agents):
            if(timestep % training_interval==0):                
                for _ in range(train_steps):
                    agent.train(agnt)

        states = next_state
        rewards += reward
        if(np.any(done)):
            break
    rewards = max(rewards)
    scores.append(rewards)
    scores_last_hundred_episodes.append(rewards)
    #if(episode % 100 == 0):
    print('episode {} frames {} rewards {:.2f} mean score(100ep) {:.2f}'.format(episode, timesteps, rewards, np.mean(scores_last_hundred_episodes)))
torch.save(agent.networks[0].actor.state_dict(), 'agent1_actor_checkpoint.pth')
torch.save(agent.networks[0].critic.state_dict(), 'agent1_critic_checkpoint.pth')
torch.save(agent.networks[1].actor.state_dict(), 'agent2_actor_checkpoint.pth')
torch.save(agent.networks[1].critic.state_dict(), 'agent2_critic_checkpoint.pth')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()



# noise scaling
# clipping?