## Multi Agent Deep Deterministic Policy gradients
Uses fixed q-targets and experience replay
2 agent environment with neural networks, one network each for Actor, one network each for Critic.
Three layer fully connected neural networks with batch normalization, relu and tanh activation.
The critic takes in the actions selected by the actors.

#### Hyperparameters
actor learning rate 0.0001

critic learning rate 0.001

stores the last 1 million experience tuples in the replay buffer

discounts future rewards at rate gamma 0.95

runs 128 sample SARS tuples through the network for every training run, batch size 128     

copies weights to target networks after every training run at rate tau 0.001

training_interval 4, trains the networks every 4 timesteps

train_steps 2, trains the networks two times for every agent 	

#### Results
[image1]: https://github.com/Maggern3/MADDPG/blob/master/training_results.png "Trained Agent"
[image2]: https://github.com/Maggern3/MADDPG/blob/master/training_results_500.png "Trained Agent last 500 episodes"

![Trained Agent][image1]
![Trained Agent last 500 episodes][image2]

The environment was solved in 4609 episodes.

#### Ideas for future work
Using dropout, adding prioritized experience replay, noise scaling(decrease noise as training progress), further tuning of hyperparameters
