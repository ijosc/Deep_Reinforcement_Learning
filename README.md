# Deep Reinforcement Learning
This is the python code for my master thesis titled 'Deep Reinforcement Learning For Arcade Games' at Denmarks Technical University.

My thesis is inspired by this DeepMind article http://arxiv.org/abs/1312.5602. My aim is to first replicate the article and use the results as a baseline. Subsequently I will attempt to improve the final score, the learning rate and/or the generalisability by: 
- Using a range for reward instead of fixing them to -1,0,1
- Implementing prioritised sweeping for the experience replay
- Experimenting with different CNN architectures (filter sizes, pooling layers, hyper parameters, etc.)
- Implementing and experimenting with the saddle-free newton method (http://arxiv.org/abs/1406.2572)
- Experimenting with different games and investigating the applicability of various setups/architectures for transfer learning
- Implementing RNN layers and experimenting with various configurations for final layers

The tools I am using are
[Deeppy](https://github.com/andersbll/deeppy),
[CUDArray](https://github.com/andersbll/cudarray) and
[Arcade Learning Environment](http://www.arcadelearningenvironment.org/).

My code is based on [Replicating DeepMind](https://github.com/kristjankorjus/Replicating-DeepMind).

I have switched to Theano half-way through the project. This repository is therefore not up-to-date. My Theano implementation can be found here: [Deep Reinforcement Learning with Theano](https://github.com/ijosc/theano_deep_rl)
