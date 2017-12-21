#/home/peiran/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Implementation of Siamese LSTM for similarity of text sentences
"""
import tensorflow as tf
import numpy as np
import math

class Config:
    """Config class is for storing the information about the models
    and data set. Model objects are passed a Config() object at
    instantiation.
    """
    def __init__(self):
        self.max_length = 30 # longest sequence to parse
        self.n_label = 1
        self.dropout = 0.5
        self.embed_size = 300
        self.hidden_size = 300
        self.minibatch_size = 64
        self.n_epochs = 10
        self.learning_rate = 0.0001
        self.model_name = "Siamese_LSTM"
        self.seed = 1
        self.print_cost = True

class siamese_lstm():
    """
    Implements a siamese lstm neural network with an embedding layer
    This network will predict the similarity of two queries, represented by 
    two indexed and padded sequences
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training. 
        """
        self.input_q1_placeholder = tf.placeholder(tf.int32, \
                (None, self.Config.max_length), name="question1") # sentence sequence
        self.input_q2_placeholder = tf.placeholder(tf.int32, \
                (None, self.Config.max_length), name="question2") # sentence sequence
        self.labels_placeholder = tf.placeholder(tf.float64, \
                (None, self.Config.n_label), name="label") # labels
        
    def create_feed_dict(self, minibatch_X_q1, minibatch_X_q2, minibatch_Y):
        """Creates the feed_dict for the dependency parser.
        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }
        """
        feed_dict={self.input_q1_placeholder: minibatch_X_q1, \
                   self.input_q2_placeholder: minibatch_X_q2, \
                   self.labels_placeholder: minibatch_Y}
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        embedding_mat = tf.Variable(self.pre_embeddings, trainable=False)

        embeddings_q1 = tf.nn.embedding_lookup(
                            embedding_mat,
                            self.input_q1_placeholder)   

        embeddings_q2 = tf.nn.embedding_lookup(
                            embedding_mat,
                            self.input_q2_placeholder)    
        return embeddings_q1, embeddings_q2

    def add_feedforward_op(self):
        """Adds the Siamese RNN, feed in the input data and get the prediction
        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """
        embeddings_q1, embeddings_q2 = self.add_embedding()

        # define the LSTM cell
        def siamese_lstm_bk(q1, q2):
            q1_reshape = tf.unstack(q1, self.Config.max_length, 1)
            q2_reshape = tf.unstack(q2, self.Config.max_length, 1)

            with tf.variable_scope("siamese_lstm"):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.Config.hidden_size, forget_bias=1.0)

            outputs_1, states_1 = tf.nn.static_rnn(lstm_cell, q1_reshape, dtype=tf.float64)
            outputs_2, states_2 = tf.nn.static_rnn(lstm_cell, q2_reshape, dtype=tf.float64)
            return outputs_1[-1], outputs_2[-1]

        out_1, out_2 = siamese_lstm_bk(embeddings_q1, embeddings_q2)
        sim_measure = tf.exp(-tf.reduce_sum(tf.abs(out_1-out_2), axis=1, keep_dims=True), name="sim_measure")

        return sim_measure

    def add_cost_op(self, similarity):
        """Adds Ops for the loss function to the computational graph.
        Args:
            similarity: A tensor containing the sim_measure
        """
        # calculate the cost function
        cost = tf.reduce_mean(tf.square(similarity - self.labels_placeholder))
        return cost
    
    def add_prediction_accuracy(self, similarity):
        correct_prediction = tf.equal(tf.round(similarity), tf.round(self.labels_placeholder))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        return accuracy

    def add_training_op(self, cost):
        """Sets up the training Ops.
        Creates an optimizer and applies the gradients to all trainable variables.
        Args:
            cost: cost tensor
        Returns:
            optimizer: The Op for training.
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.Config.learning_rate).minimize(cost)
        return optimizer
    
    def add_init(self):
        init = tf.global_variables_initializer()
        return init
    
    def random_mini_batches(self,X_q1, X_q2, Y, mini_batch_size = 64, seed = 0):
        np.random.seed(seed)            # set the random seed
        m = Y.shape[0]                  # number of training examples
        mini_batches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X_q1 = X_q1[permutation,:]
        shuffled_X_q2 = X_q2[permutation,:]
        shuffled_Y = Y[permutation,:]

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        # number of mini batches of size mini_batch_size in the partitionning
        num_complete_minibatches = math.floor(m/mini_batch_size) 
        for k in range(0, num_complete_minibatches):
            mini_batch_X_q1 = shuffled_X_q1[k*mini_batch_size:(k+1)*mini_batch_size,:]
            mini_batch_X_q2 = shuffled_X_q2[k*mini_batch_size:(k+1)*mini_batch_size,:]
            mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size,:]
            mini_batch = (mini_batch_X_q1, mini_batch_X_q2, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X_q1 = shuffled_X_q1[-(m % mini_batch_size):,:]
            mini_batch_X_q2 = shuffled_X_q2[-(m % mini_batch_size):,:]
            mini_batch_Y = shuffled_Y[-(m % mini_batch_size):,:]
            mini_batch = (mini_batch_X_q1, mini_batch_X_q2, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches
    
    def fit(self, train_q1, train_q2, Y_label):
        print("Creating graph...")
        tf.reset_default_graph()
        self.add_placeholders()
        sim_measure = self.add_feedforward_op()
        cost = self.add_cost_op(sim_measure)
        optimizer = self.add_training_op(cost)
        accuracy = self.add_prediction_accuracy(sim_measure)
        init = self.add_init()
        seed = self.Config.seed
        m = Y_label.shape[0]
        print("Graph is ready")
        
        # a container of the cost value after each iteration
        self.costs = []
        
        print("Exucute graph...")
        with tf.Session() as sess:
            
            # Run the initialization
            sess.run(init)

            # Do the training loop
            for epoch in range(self.Config.n_epochs):

                # Defines a cost related to each epoch
                epoch_cost = 0. 

                # update random seed each epoch
                seed = seed + 1

                # number of minibatches of size minibatch_size in the train set
                num_minibatches = int(m / self.Config.minibatch_size) 

                # use random_mini_batches to split mini batches
                minibatches = self.random_mini_batches(train_q1, train_q2, Y_label, \
                                                  self.Config.minibatch_size, self.Config.seed)

                # iterate on each mini-batch
                for minibatch in minibatches:
                    #f a minibatch
                    (minibatch_X_q1, minibatch_X_q2, minibatch_Y) = minibatch

                    # feed the mini-batch to the graph for execution
                    feed = self.create_feed_dict(minibatch_X_q1, minibatch_X_q2, minibatch_Y)
                    minibatch_cost, _ = sess.run([cost, optimizer], feed)
                    epoch_cost += minibatch_cost / num_minibatches

                # Print the cost every epoch
                if self.Config.print_cost and epoch % 1 == 0:
                    feed = self.create_feed_dict(train_q1, train_q2, Y_label)
                    print("After epoch %i --- cost: %f, accuracy: %f" % 
                          (epoch, epoch_cost, sess.run(accuracy,feed))) 
                if epoch % 5 == 0:
                    self.costs.append(epoch_cost)

    def __init__(self, config, pretrained_embeddings):
        self.Config = Config()
        self.pre_embeddings = pretrained_embeddings

if __name__ == "__main__":

    INPUT_DIR = "./input/"

    train_pad_q1 = np.load(INPUT_DIR+"train_pad_q1.npy")
    train_pad_q2 = np.load(INPUT_DIR+"train_pad_q2.npy")
    Y_train_pad = np.load(INPUT_DIR+"Y_train.npy")
    Y_train_pad = Y_train_pad.reshape(-1,1)
    pretrained_embeddings = np.load(INPUT_DIR+"embedding.npy")

    model = siamese_lstm(Config, pretrained_embeddings)
    model.fit(train_pad_q1,train_pad_q2,Y_train_pad)



