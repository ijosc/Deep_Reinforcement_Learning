import deeppy as dp

class NeuralNet():

    def __init__(self, nr_inputs, nr_outputs, net):
        """
        Initialize a NeuralNet

        @param nr_inputs: number of inputs in data layer
        @param nr_outputs: number of target values in another data layer
        """

        # Save data parameters
        self.nr_inputs = nr_inputs
        self.nr_outputs = nr_outputs
        

    def train(self, inputs, outputs):
        """
        Train neural net with inputs and outputs.

        @param inputs: NxM numpy.ndarray, where N is number of inputs and M is batch size
        @param outputs: KxM numpy.ndarray, where K is number of outputs and M is batch size
        @return cost?
        """

        def val_error():
                return self.error(test_input)
        n_epochs = [8, 8]
        learn_rate = 0.001
        for i, max_epochs in enumerate(n_epochs):
            lr = learn_rate/10**i
            trainer = dp.StochasticGradientDescent(
                max_epochs=max_epochs,
                learn_rule=dp.Momentum(learn_rate=lr, momentum=0.9),
            )
            trainer.train(self, train_input, val_error)

    def predict(self, inputs):
        """
        Predict neural network output layer activations for input.

        @param inputs: NxM numpy.ndarray, where N is number of inputs and M is batch size
        """
        assert inputs.shape[0] == self.nr_inputs

        batch_size = inputs.shape[1]
        outputs = np.zeros((batch_size, self.nr_outputs), dtype=np.float32)

        # start feed-forward pass in GPU
        self.libmodel.startFeatureWriter([inputs, outputs.transpose().copy()], [outputs], [self.output_layer_name])
        # wait until processing has finished
        self.libmodel.finishBatch()

        # now activations of output layer should be in 'outputs'
        return outputs

    # def get_weight_stats(self):
    #     # copy weights from GPU to CPU memory
    #     self.sync_with_host()
    #     wscales = OrderedDict()
    #     for name,val in sorted(self.layers.items(), key=lambda x: x[1]['id']): # This is kind of hacky but will do for now.
    #         l = self.layers[name]
    #         if 'weights' in l:
    #             wscales[l['name'], 'biases'] = (n.mean(n.abs(l['biases'])), n.mean(n.abs(l['biasesInc'])))
    #             for i,(w,wi) in enumerate(zip(l['weights'],l['weightsInc'])):
    #                 wscales[l['name'], 'weights' + str(i)] = (n.mean(n.abs(w)), n.mean(n.abs(wi)))
    #     return wscales

    def save_network(self, epoch):
    	pass