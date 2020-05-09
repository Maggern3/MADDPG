## MADDPG
#### Project environment
8 states for each agent, stacked 3 times, yielding an observation of 24 dimensions, there's 2 agents collaborating. Each agent has 2 continous actions.
The environment is considered solved when an average of a max score between the agents of 0.5 is maintained for 100 episodes.

#### Installation 
First clone the [udacity deep reinforcement learning repo](https://github.com/udacity/deep-reinforcement-learning) 
and navigate to it's directory then
```
cd python
pip install .
```
this installs the required dependencies. 

Download the Tennis Unity-ML environment from one of the following links:

Windows: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
     
Mac: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
     
Put it in the root of the cloned project folder and unzip to Tennis_Windows_x86_64 folder. 

You should now be able to build and run the project.

#### How to train the agent
Run the following command to train the agent
```
python runner.py
```