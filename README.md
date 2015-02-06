# Deep Reinforcement Learning
This is the python code for my master thesis titled'Deep Reinforcement Learning For Arcade Games' at Denmarks Technical University.

My thesis is inspired by this DeepMind article http://arxiv.org/abs/1312.5602. My aim is to first replicate the article and use the results as a baseline. Subsequently I will attempt to improve the final score, the learning rate and/or the generalisability by: 
- Using a range for reward instead of fixing them to -1,0,1
- Implementing prioritised sweeping for the experience replay
- Experimenting with different CNN architectures (filter sizes, pooling layers, hyper parameters, etc.)
- Implementing and experimenting with the saddle-free newton method (http://arxiv.org/abs/1406.2572)
- Experimenting with different games and investigate the applicability of this method for transfer learning
- Implement RNN layers and experiment with various configurations for final layers
