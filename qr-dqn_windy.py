#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 08:04:24 2018

@author: sannebjartmar
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
from common.replay_buffer import ReplayBuffer
from risk_strategies import behaviour_policy
from wind_world import *
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)

env = WindyGridworldEnv()
env.reset()

dists = [0]
stats = [0]

class QRDQN(nn.Module):
    # nn.Module = Base class for all neural network modules
    def __init__(self, num_inputs, num_actions, num_quants):
        # num_inputs = observation space - env.observation_space
        # num_actions = action that the agent can take - env.action_space
        # num_quants = number of quantiles in distribution 
        super(QRDQN, self).__init__()
        
        self.num_inputs  = num_inputs
        self.num_actions = num_actions
        self.num_quants  = num_quants
        
        # Not standard DQN since the input are the features not image 
        # --> Can be translated to gridworld with just the coordinates?
        # Is this "standard implementation" of non image-input?
        self.features = nn.Sequential(
            nn.Linear(num_inputs, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 80),
            nn.ReLU(),
            nn.Linear(80, self.num_actions * self.num_quants) 
        )
        
    def forward(self, x):
        # Forward pass
        batch_size = x.size(0)
        x = self.features(x)
        # Changes the dimensions to batch size x actions x quantiles in distribution
        x = x.view(batch_size, self.num_actions, self.num_quants)
        return x
        
    def act(self, state, epsilon, behaviour_p):
        # Exploration 
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.array(state, dtype=np.float32)).unsqueeze(0), volatile=True)
            x = self.forward(state)
            d = x
            d = d.detach().numpy()
            
            dists[0] = d
            s = behaviour_policy(d, behaviour_p, quant = 0.95, const = 1)
            stats[0] = s
            
            s = torch.tensor(s)
            s = s.unsqueeze(0)
            
            evaluation = s
            action  = evaluation.max(1)[1] # max nr dim 1, index 0 for value and index 1 for argmax
            action  = action.data.numpy()[0]
        else:
            action = random.randrange(self.num_actions)
        return action
    
def projection_distribution(dist, gamma, next_state, reward, done):
    next_dist = target_model(next_state) # batch size x actions x quantiles
    
    next_action = next_dist.mean(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_quant)
    next_dist = next_dist.gather(1, next_action).squeeze(1).data # batch size x quantiles  

    expected_quant = reward.unsqueeze(1) + gamma * next_dist * (1 - done.unsqueeze(1))
    expected_quant = Variable(expected_quant)
        
    # sort quantile values
    quant_idx = torch.sort(expected_quant, 1, descending=False)[1]
    quant_idx = torch.sort(quant_idx, 1, descending=False)[1].numpy()

    tau_hat = quant_idx + 1
    tau_hat = tau_hat/num_quant
    tau_hat = tau_hat - 0.5/num_quant
    tau_hat = torch.tensor(tau_hat.astype(dtype=np.float32))
    return tau_hat, expected_quant

def compute_td_loss(batch_size, gamma):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size) 

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(np.float32(done))
    
    dist = current_model(state)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_quant)
    dist = dist.gather(1, action).squeeze(1)
    
    tau, expected_quant = projection_distribution(dist, gamma, next_state, reward, done)
    k = 1
    
    u = expected_quant - dist
    huber_loss = 0.5 * u.abs().clamp(min=0.0, max=k).pow(2)
    huber_loss += k * (u.abs() - u.abs().clamp(min=0.0, max=k))
    quantile_loss = (autograd.Variable(tau) - ((u < 0).float())).abs() * (huber_loss)
    loss = (quantile_loss.sum() / num_quant) # average loss over batch
        
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm(current_model.parameters(), 0.5) 
    optimizer.step()
    
    return loss


num_quant = 30
current_model = QRDQN(2, env.action_space.n, num_quant)
target_model  = QRDQN(2, env.action_space.n, num_quant)
    
optimizer = optim.Adam(current_model.parameters())

replay_buffer = ReplayBuffer(10000)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
    
update_target(current_model, target_model)   

epsilon_start = 0.1
epsilon_min = 0.01
lr_start = 0.1
num_episodes = 20000 
batch_size = 50
gamma      = 1

lr_by_episode = lambda episode_idx: lr_start / (2**(int(lr_start/2000)))
epsilon_by_episode = lambda episode_idx: max(epsilon_start / (2**(int(episode_idx/2000))), epsilon_min)
losses = []
all_rewards = []
episode_reward = 0
behaviour_p = "mean" #mean, sharp ratio, sortino sharp ratio, weigthed VaR, weigthed cVaR

state = env.reset()
for episode_idx in range(1, num_episodes + 1):
    
    if episode_idx < batch_size:
        action = random.randrange(env.action_space.n)
    else:
        action = current_model.act(state, epsilon_by_episode(episode_idx), behaviour_p)
    
    next_state, reward, done, _ = env.step(action)  
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size, gamma)
        losses.append(loss.data[0])
        
    if episode_idx % 1000 == 0:
        update_target(current_model, target_model) # save model
        
        # Save data and model parameters
        np.save("results/cart_rewards_%s_%s" %(behaviour_p, episode_idx), all_rewards)
        np.save("results/cart_losses_%s_%s" %(behaviour_p, episode_idx), losses)
        np.save("results/cart_stat_%s_%s" %(behaviour_p, episode_idx), stats)
        np.save("results/cart_dists_%s_%s" %(behaviour_p, episode_idx), dists)
        torch.save(current_model.state_dict(), "results/cart_model_%s_%s" %(behaviour_p, episode_idx))
        
