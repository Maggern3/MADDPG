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

training_interval = 4 #20
train_steps = 2 #1
agent = MultiAgent(state_size, action_space_size, agents)
scores = []
scores_last_hundred_episodes = deque(maxlen=100)
mean_hundred_scores = []
actions = np.random.randn(2, 2)
actions = np.clip(actions, -1, 1)  
#print(actions)
total_frames = 0
for episode in range(4800): # 1000-30k
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
    total_frames += timesteps
    rewards = max(rewards)
    scores.append(rewards)
    scores_last_hundred_episodes.append(rewards)
    mean_hundred_score = np.mean(scores_last_hundred_episodes)
    mean_hundred_scores.append(mean_hundred_score)
    #if(episode % 100 == 0):
    print('episode {} frames {} rewards {:.2f} mean score(100ep) {:.2f} total frames {}'.format(episode, timesteps, rewards, mean_hundred_score, total_frames))
torch.save(agent.networks[0].actor.state_dict(), 'agent1_actor_checkpoint.pth')
torch.save(agent.networks[0].critic.state_dict(), 'agent1_critic_checkpoint.pth')
torch.save(agent.networks[1].actor.state_dict(), 'agent2_actor_checkpoint.pth')
torch.save(agent.networks[1].critic.state_dict(), 'agent2_critic_checkpoint.pth')

version = 'v1'
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('results/{}/scores.png'.format(version))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(500), scores[-500:])
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('results/{}/scores_500.png'.format(version))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(mean_hundred_scores)), mean_hundred_scores)
plt.ylabel('Mean 100 Score')
plt.xlabel('Episode #')
plt.savefig('results/{}/mean_scores.png'.format(version))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(500), mean_hundred_scores[-500:])
plt.ylabel('Mean 100 Score')
plt.xlabel('Episode #')
plt.savefig('results/{}/mean_scores_500.png'.format(version))


# noise scaling
# increase gamma?
# training interval and steps
# reduce buffer size, to not hold 1000 episodes +, say 3-500 episodes

# canonical
#episode 4572 frames 660 rewards 1.70 mean score(100ep) 0.85
#episode 4999 frames 375 rewards 1.00 mean score(100ep) 0.47

# 512, sigma 0.1, batch size 256
#episode 4497 frames 108 rewards 0.30 mean score(100ep) 0.72
#episode 4999 frames 89 rewards 0.20 mean score(100ep) 0.12

#machine1 canonical + 256
#episode 4709 frames 1001 rewards 2.60 mean score(100ep) 0.50
#episode 4798 frames 727 rewards 1.90 mean score(100ep) 1.12

#machine2 canonical + 512

#train_steps = 1

# fc 64, 128

# buffer size 200k

# set training interval 100 on plateau timesteps 1001