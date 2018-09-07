# Risk averse reinforcement learning. 

This code combines recent progress in Distributional Reinforcement Learning with Quantile Regression (https://arxiv.org/pdf/1710.10044.pdf) with classic theory to create risk averse, safe algorithm.

The cartpole environment is implemented using OpenAI Gym. This is a toolkit for developing and comparing reinforcement learning algorithms. Installation guidelines and other information can be found at https://github.com/openai/gym. 

We use Pytorch to build the networks, for installation and other information please see https://pytorch.org/. 

The common folder contains helper functions and classes: layers, replay buffert and wrappers. They are all from Open AI Baselines (https://github.com/openai/baselines). 

The risk averse strategies can be found in risk_strategies.py. wind_world.py and contains the Windy World environment. 

qr-dqn_cart and qr-dqn_windy contains the implementations themselves, in Cartpole Environment and Windy Gridworld environment.  
To download:

'''
git clone https://github.com/sannebh/riskaverse_rl/
'''
