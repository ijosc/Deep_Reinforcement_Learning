from utils.ale_interface import ALE
from memory.memoryd import MemoryD
from ai.deepmind_net import net as net

import random
import numpy as np
import time
import sys
import cPickle as pickle

import deeppy as dp

from os import linesep as NL

class Main(object):
    """
    Main class for starting training and testing
   """
 
    # How many transitions to keep in memory?
    memory_size = 500000

    # Size of the mini-batch, 32 was given in the paper, popular choice is 128
    minibatch_size = 32

    # Number of possible actions in a given game, 6 for "Breakout"
    number_of_actions = 6

    # Size of one frame
    frame_size = 84*84

    state_length = 4 

    # Size of one state is four 84x84 screens
    state_size = state_length * frame_size

    # Discount factor for future rewards
    discount_factor = 0.9

    # Exploration rate annealing speed
    epsilon_frames = 1000000.0

    # Epsilon during testing
    test_epsilon = 0.05

    # Total frames played, only incremented during training
    total_frames_trained = 0

    # Number of random states to use for calculating Q-values
    nr_random_states = 100

    # Random states that we use to calculate Q-values
    random_states = None

    # Memory itself
    memory = None

    # Neural net
    nnet = None

    # Communication with ALE
    ale = None

    # The last 4 frames the system has seen
    current_state = None    

    def __init__(self):
        self.memory = MemoryD(self.memory_size)
        self.ale = ALE(display_screen="false", skip_frames=4, game_ROM='ale/roms/breakout.bin')
        self.nnet = net
        self.trainer =dp.StochasticGradientDescent(
            max_epochs=10,
            learn_rule=dp.Momentum(learn_rate=0.001, momentum=0.9),
        )
        self.q_values = []

    def compute_epsilon(self, frames_played):
        """
        From the paper: "The behavior policy during training was epsilon-greedy
        with annealed linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter."
        @param frames_played: How far are we with our learning?
        """
        return max(0.99 - frames_played / self.epsilon_frames, 0.1)

    def predict_best_action(self, last_state):
        # use neural net to predict Q-values for all actions
        qvalues = self.nnet.predict(last_state)
        print "Predicted action Q-values: ", qvalues
        self.q_values.append(np.max(qvalues))

        # return action (index) with maximum Q-value
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
        for i, action in enumerate(actions):
            qvalues[i][action] = rewards[i] + self.discount_factor * max_qvalues[i]

        train_input = dp.SupervisedInput(prestates.x, qvalues, batch_size=32)
        self.trainer.train(net, train_input)

        error = self.nnet.error(train_input)
        print "error: " + str(error)

        return error
        
    def play_games(self, nr_frames, epoch, train, epsilon = None):
        """
        Main cycle: starts a game and plays number of frames.
        @param nr_frames: total number of games allowed to play
        @param train: true or false, whether to do training or not
        @param epsilon: fixed epsilon, only used when not training
        """
        # assert train or epsilon is not None

        frames_played = 0
        game_scores = []

        # Start a new game
        first_frame = self.ale.new_game()
        if train:
            self.memory.add_first(first_frame)
        else:  # if testing we dont write anything to memory
            pass

        # We need to initialize/update the current state
        if self.current_state == None:
            print "current state is none"
            self.current_state = np.empty((1, self.state_length, 84, 84), dtype=np.float64)
            for i in range(self.state_length):
                self.current_state[0,i,:,:] = first_frame.copy()
        else:
            for i in range(self.state_length):
                if i<3:
                    self.current_state.x[0,i,:,:] = self.current_state.x[0,i+1,:,:]
                else:
                    self.current_state.x[0,i,:,:] = first_frame.copy()

        game_score = 0
        if train and epoch==1:
            self.current_state = dp.Input(self.current_state)
            self.current_state.y_shape=(1,6)
            self.nnet._setup(self.current_state)

        # Play games until maximum number is reached
        while frames_played < nr_frames:

            # Epsilon decreases over time only when training
            if train:
                epsilon = self.compute_epsilon(self.total_frames_trained)
                print "Current annealed epsilon is %f at %d frames" % (epsilon, self.total_frames_trained)

            # Some times random action is chosen 
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(self.number_of_actions))
                print "Chose random action %d" % action
            # Usually neural net chooses the best action
            else:
                action = self.predict_best_action(self.current_state)
                print "Neural net chose action %d" % int(action)

            # Make the move. Returns points received and the new state
            points, next_frame = self.ale.move(action)

            # Changing points to rewards
            if points > 0:
                print "    Got %d points" % points
                reward = 1
            else:
                reward = 0

            # Book keeping
            game_score += points
            frames_played += 1
            #print "Played frame %d" % frames_played

            # We need to update the current state

            for i in range(self.state_length):
                if i<3:
                    self.current_state.x[0,i,:,:] = self.current_state.x[0,i+1,:,:]
                else:
                    self.current_state.x[0,i,:,:] = next_frame

            # Only if training
            if train:

                # Store new information to memory
                self.memory.add(action, reward, next_frame)

                # Increase total frames only when training
                self.total_frames_trained += 1

                # Fetch random minibatch from memory
                minibatch = self.memory.get_minibatch(self.minibatch_size)

                # Train neural net with the minibatch
                self.train_minibatch(minibatch)
                #print "Trained minibatch of size %d" % self.minibatch_size

            # Play until game is over
            if self.ale.game_over:
                print "    Game over, score = %d" % game_score
                # After "game over" increase the number of games played
                game_scores.append(game_score)
                game_score = 0

                # And do stuff after end game
                self.ale.end_game()
                if train:
                    self.memory.add_last()
                else:
                    pass
                first_frame = self.ale.new_game()
                if train:
                    self.memory.add_first(first_frame)
                else:  # if testing we dont write anything to memory
                    pass

                # We need to update the current state
                for i in range(self.state_length):
                    if i<3:
                        self.current_state.x[0,i,:,:] = self.current_state.x[0,i+1,:,:]
                    else:
                        self.current_state.x[0,i,:,:] = first_frame.copy()


        # reset the game just in case
        self.ale.end_game()

        return game_scores
    def run(self, epochs, training_frames, testing_frames):

        for epoch in range(1, epochs + 1):
            print "Epoch %d:" % epoch

            if training_frames > 0:
                # play number of frames with training and epsilon annealing
                print "  Training for %d frames" % training_frames
                training_scores = self.play_games(training_frames, epoch, train = True)


            if testing_frames > 0:
                # play number of frames without training and without epsilon annealing
                print "  Testing for %d frames" % testing_frames
                testing_scores = self.play_games(testing_frames, epoch, train = False, epsilon = self.test_epsilon)

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
        pickle.dump([self.q_values,testing_scores], open("q_values.p", "wb" ))
        pickle.dump(self.nnet, open("nnet.p", "wb" ))

if __name__ == '__main__':
    # take some parameters from command line, otherwise use defaults
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    training_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    testing_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    m = Main()
    m.run(epochs, training_frames, testing_frames)