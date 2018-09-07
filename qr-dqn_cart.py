#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 07:54:24 2018

@author: sannebjartmar
"""


import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
from common.replay_buffer import ReplayBuffer
from risk_strategies import behaviour_policy



Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)

env_id = "CartPole-v0" 
# Observations: cart position, cart velocity, pole angle, pole velocity at tip
env = gym.make(env_id)

dists = [0]
stats = [0]

class QRDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_quants):
        super(QRDQN, self).__init__()
        
        self.num_inputs  = num_inputs
        self.num_actions = num_actions
        self.num_quants  = num_quants
    
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
        batch_size = x.size(0)
        x = self.features(x)
        x = x.view(batch_size, self.num_actions, self.num_quants)
        return x
        
    def act(self, state, epsilon, behaviour_p):
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
            action  = evaluation.max(1)[1] 
            action  = action.data.numpy()[0]
        else:
            action = random.randrange(self.num_actions)
        return action
    
def projection_distribution(dist, gamma, next_state, reward, done, epsilon):
    next_dist = target_model(next_state)
    
    # Q-learning 
    next_action = next_dist.mean(2).max(1)[1]
    
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_quant)
    next_dist = next_dist.gather(1, next_action).squeeze(1).data 
    
    expected_quant = reward.unsqueeze(1) + gamma * next_dist * (1 - done.unsqueeze(1))
    expected_quant = Variable(expected_quant)
    
    # Sort quantile values
    quant_idx = torch.sort(expected_quant, 1, descending=False)[1]
    quant_idx = torch.sort(quant_idx, 1, descending=False)[1].numpy()

    tau_hat = quant_idx + 1
    tau_hat = tau_hat/num_quant
    tau_hat = tau_hat - 0.5/num_quant
    tau_hat = torch.tensor(tau_hat.astype(dtype=np.float32))
    return tau_hat, expected_quant

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
    
def compute_td_loss(batch_size, gamma, epsilon):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size) 

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(np.float32(done))

    dist = current_model(state)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_quant)
    dist = dist.gather(1, action).squeeze(1)
    
    tau, expected_quant = projection_distribution(dist, gamma, next_state, reward, done, epsilon)
    k = 1
    
    u = expected_quant - dist
    huber_loss = 0.5 * u.abs().clamp(min=0.0, max=k).pow(2)
    huber_loss += k * (u.abs() - u.abs().clamp(min=0.0, max=k))
    quantile_loss = (autograd.Variable(tau) - ((u < 0).float())).abs() * (huber_loss)
    loss = (quantile_loss.sum() / num_quant)
        
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm(current_model.parameters(), 0.5) 
    optimizer.step()
    
    return loss

epsilon_start = 0.1
epsilon_min = 0.01
num_episodes = 10000
batch_size = 50
gamma      = 1

epsilon_by_episode = lambda episode_idx: max(epsilon_start / (2**(int(episode_idx/2000))), epsilon_min)
losses = []
all_rewards = []
episode_reward = 0
behaviour_p = "mean" # Alternatives: mean, sharp ratio, sortino sharp ratio, weigthed VaR, weigthed cVaR
noisy = False

num_quant = 30
current_model = QRDQN(env.observation_space.shape[0], env.action_space.n, num_quant)
target_model  = QRDQN(env.observation_space.shape[0], env.action_space.n, num_quant)
update_target(current_model, target_model)
    
optimizer = optim.Adam(current_model.parameters())
replay_buffer = ReplayBuffer(10000)
state = env.reset()

for episode_idx in range(1, num_episodes + 1):
    
    if episode_idx < batch_size:
        action = random.randrange(env.action_space.n)
    else:
        action = current_model.act(state, epsilon_by_episode(episode_idx), behaviour_p)

    next_state, reward, done, _ = env.step(action)
    
    if noisy:
        reward += np.random.normal(0, 0.1)
    
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size, gamma, epsilon_by_episode(episode_idx))
        losses.append(loss.data[0])
        
    if episode_idx % 1000 == 0:
        update_target(current_model, target_model) # Update target
        
        # Save data and model parameters
        np.save("/Users/sannebjartmar/Desktop/Thesis/results/cart_rewards_%s_%s" %(behaviour_p, episode_idx), all_rewards)
        np.save("/Users/sannebjartmar/Desktop/Thesis/results/cart_losses_%s_%s" %(behaviour_p, episode_idx), all_rewards)
        np.save("/Users/sannebjartmar/Desktop/Thesis/results/cart_stat_%s_%s" %(behaviour_p, episode_idx), all_rewards)
        np.save("/Users/sannebjartmar/Desktop/Thesis/results/cart_dists_%s_%s" %(behaviour_p, episode_idx), all_rewards)
        torch.save(current_model.state_dict(), "/Users/sannebjartmar/Desktop/Thesis/results/cart_model_%s_%s" %(behaviour_p, episode_idx))
        
        
   