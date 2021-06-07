import torch
import numpy as np
import pandas as pd
from collections import deque
from unityagents import UnityEnvironment
import random
import matplotlib.pyplot as plt
# %matplotlib inline

from ddpg_agent import Agent

def plot_scores(scores, rolling_window=10, save_fig=False):
    """Plot scores and optional rolling mean using specified window."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(f'scores')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean)

    if save_fig:
        plt.savefig(f'figures_scores.png', bbox_inches='tight', pad_inches=0)
        
def ddpg(num_agents, n_episodes=1000, max_t=200, print_every=2):
    
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
                        
            rewards = env_info.rewards                         # get reward (for each agent)
            rewards = [0.1 if rew > 0 else 0 for rew in rewards]
            done = (env_info.local_done)                     # see if episode finished
            
            if num_agents > 1:
                for i in range(num_agents):
                    agent.step(states[i], actions[i], rewards[i], next_states[i], done[i])
            else:
                agent.step(states, actions, rewards, next_states, done)
            
            scores += rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(done):                                  # exit loop if episode finished
                break
        scores_deque.append(score)
        scores.append(score)
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            # plot_scores(scores)
        
        if np.mean(scores_deque) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), '../weights/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), '../weights/checkpoint_critic.pth')
            break
    return scores

# select this option to load version 1 (with a single agent) of the environment
# env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')

env = UnityEnvironment(file_name='./Reacher_Linux/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')
# env = UnityEnvironment(file_name='./Reacher_Linux/Reacher_Linux_NoVis/Reacher.x86_64')

# select this option to load version 2 (with 20 agents) of the environment
# env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = Agent(state_size=state_size, action_size=action_size,
              num_agents=num_agents, random_seed=42)

scores = ddpg(num_agents)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()
