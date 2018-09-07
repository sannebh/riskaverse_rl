# Risk averse reinforcement learning. 

This code combines recent progress in distributional reinforcement learning (https://arxiv.org/pdf/1710.10044.pdf) with classic theory to create risk averse, safe algorithm built in Pytorch.  

Common contains helper functions and classes: layers, replay buffert and wrappers. They are all from Open AI Baselines (https://github.com/openai/baselines). 

qr-dqn_cart and qr-dqn_windy contains the implementations themselves, in Cartpole Environment and Windy Gridworld environment. 
