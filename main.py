from utils.ale_interface import ALE
from memory.memoryd import MemoryD
from utils.game_actions import action_dict
from ai.deepmind_net import net as net

import random
import numpy as np
import sys
import cPickle as pickle

import deeppy as dp
import os

class Main(object):
    """
    Main class for starting training and testing
   """

    memory_size = 500000
    memory = None

    minibatch_size = 32

    frame_size = 84*84

    state_length = 4 
    state_size = state_length * frame_size

    discount_factor = 0.9
    epsilon_frames = 1000000.0
    test_epsilon = 0.05

    total_frames_trained = 0
    nr_random_states = 100
    random_states = None

    nnet = None
    ale = None

    current_state = None    

    def __init__(self, game_name, run_id):

        self.number_of_actions = len(action_dict[game_name])
        valid_actions = action_dict[game_name]

        net.layers[-2] = dp.FullyConnected(n_output=self.number_of_actions,
            weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                 weight_decay=0.004, 
                                 monitor=False))

        self.memory = MemoryD(self.memory_size)

        self.ale = ALE(valid_actions, 
            run_id, display_screen="false", 
            skip_frames=4, 
            game_ROM='ale/roms/'+game_name+'.bin')

        self.nnet = net
        self.q_values = []
        self.test_game_scores = []

    def compute_epsilon(self, frames_played):
        """
        From the paper: "The behavior policy during training was epsilon-greedy
        with annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter."
        @param frames_played: How far are we with our learning?
        """
        return max(0.99 - frames_played / self.epsilon_frames, 0.1)

    def predict_action(self, last_state, train):
        '''use neural net to predict Q-values for all actions
        return action (index) with maximum Q-value'''

        qvalues = self.nnet.predict(last_state)
        if not train: self.q_values.append(np.max(qvalues))

        return np.argmax(qvalues)

    def train_minibatch(self, minibatch):
        """
        Train function that transforms (state,action,reward,state) into (input, expected_output) for neural net
        and trains the network
        @param minibatch: list of arrays: prestates, actions, rewards, poststates
        """
        prestates, actions, rewards, poststates = minibatch
        prestates = dp.Input(prestates)

        # predict Q-values for prestates, so we can keep Q-values for other actions unchanged
        qvalues = self.nnet.predict(prestates)

        # predict Q-values for poststates
        post_qvalues = self.nnet.predict(poststates)

        # take maximum Q-value of all actions
        max_qvalues = np.max(post_qvalues, axis = 1)

        # update the Q-values for the actions we actually performed
        # remember delta value for prioritized sweeping
        for i, action in enumerate(actions):
            qvalues[i][action] = rewards[i] + self.discount_factor * max_qvalues[i]

        train_input = dp.SupervisedInput(prestates.x, qvalues, batch_size=self.minibatch_size)

        self.trainer.train(net, train_input)
        
    def play_games(self, nr_frames, epoch, train, epsilon = None):
        """
        Main cycle: starts a game and plays number of frames.
        @param nr_frames: total number of games allowed to play
        @param train: true or false, whether to do training or not
        @param epsilon: fixed epsilon, only used when not training
        """

        frames_played = 0
        game_scores = []

        first_frame = self.ale.new_game()
        if train: self.memory.add_first(first_frame)

        if self.current_state == None:
            self.current_state = np.empty((1, self.state_length, 84, 84), dtype=np.float64)
            for i in range(self.state_length):
                self.current_state[0, i, :, :] = first_frame.copy()
        else:
            self.current_state.x[0, :-1, :, :] = self.current_state.x[0, 1:, :, :] 
            self.current_state.x[0, -1, :, :] = first_frame.copy()

        game_score = 0

        if train and epoch == 1:
            self.current_state = dp.Input(self.current_state)
            self.current_state.y_shape = (1,self.number_of_actions)
            self.nnet._setup(self.current_state)

        while frames_played < nr_frames:
            if train:
                epsilon = self.compute_epsilon(self.total_frames_trained)

            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(self.number_of_actions))
            else:
                action = self.predict_action(self.current_state, train)

            points, next_frame = self.ale.move(action)

            # Changing points to rewards
            if points > 0:
                print "    Got %d points" % points
                reward = 1
            elif points < 0:
                print "    Lost %d points" % points
                reward = -1
            else:
                reward = 0

            game_score += points
            frames_played += 1

            self.current_state.x[0, :-1, :, :] = self.current_state.x[0, 1:,:,:] 
            self.current_state.x[0, -1, :, :] = next_frame

            if train:
                self.memory.add(action, reward, next_frame)
                self.total_frames_trained += 1
                minibatch = self.memory.get_minibatch(self.minibatch_size)
                self.train_minibatch(minibatch)

            if self.ale.game_over:
                print "    Game over, score = %d" % game_score
                # After "game over" increase the number of games played
                game_scores.append(game_score)
                game_score = 0

                # And do stuff after end game
                self.ale.end_game()
                if train: self.memory.add_last()

                first_frame = self.ale.new_game()
                if train: self.memory.add_first(first_frame)

                self.current_state.x[0, :-1, :, :] = self.current_state.x[0, 1:,:,:] 
                self.current_state.x[0, -1, :, :] = first_frame.copy()

        self.ale.end_game()

        return game_scores

    def run(self, epochs, training_frames, testing_frames):

        for epoch in range(1, epochs + 1):
            print "Epoch %d:" % epoch
            learn_rate = 0.0001*1/float(epoch)
            self.trainer =dp.StochasticGradientDescent(
                max_epochs=1,
                learn_rule=dp.RMSProp(learn_rate=learn_rate, decay=0.9, max_scaling=1e3),
            )
            if training_frames > 0:
                # play number of frames with training and epsilon annealing
                print "  Training for %d frames" % training_frames
                training_scores = self.play_games(training_frames, epoch, train = True)


            if testing_frames > 0:
                # play number of frames without training and without epsilon annealing
                print "  Testing for %d frames" % testing_frames
                self.test_game_scores.append(self.play_games(testing_frames, epoch, train = False, epsilon = self.test_epsilon))

                # Pick random states to calculate Q-values for
                if self.random_states is None and self.memory.count > self.nr_random_states:
                    print "  Picking %d random states for Q-values" % self.nr_random_states
                    self.random_states = self.memory.get_minibatch(self.nr_random_states)[0]

                # Do not calculate Q-values when memory is empty
                if self.random_states is not None:
                    # calculate Q-values 
                    qvalues = self.nnet.predict(self.random_states)
                    assert qvalues.shape[0] == self.nr_random_states
                    assert qvalues.shape[1] == self.number_of_actions
                    max_qvalues = np.max(qvalues, axis = 1)
                    assert max_qvalues.shape[0] == self.nr_random_states
                    assert len(max_qvalues.shape) == 1
                    avg_qvalue = np.mean(max_qvalues)
                else:
                    avg_qvalue = 0

        # save q-values to review learning progress
        # pickle.dump([self.q_values,avg_qvalue], open("q_values.p", "wb" ))
        # pickle.dump(self.nnet, open("nnet.p", "wb" ))
        # pickle.dump(self.test_game_scores, open("test_scores.p", "wb" ))

if __name__ == '__main__':
    # take some parameters from command line, otherwise use defaults
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    training_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    testing_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    run_id = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    os.remove('ale_fifo_in_%i' % run_id) if os.path.exists('ale_fifo_in_%i' % run_id) else None
    os.remove('ale_fifo_out_%i' % run_id) if os.path.exists('ale_fifo_out_%i' % run_id) else None

    m = Main('asteroids', run_id)
    m.run(epochs, training_frames, testing_frames)