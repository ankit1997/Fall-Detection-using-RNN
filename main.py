import os
import tensorflow as tf
from dataLoader import DataLoader

class FallDetector:
    def __init__(self, batch_size, num_features, timesteps, num_classes):
        self.batch_size = batch_size
        self.num_features = num_features
        self.timesteps = timesteps
        self.num_classes = num_classes

        # Model checkpoint file
        model_dir="saved_model"
        os.makedirs(model_dir, exist_ok=True)
        self.model_file = os.path.join(model_dir, "model.ckpt")

        self.build_model()
        self.define_loss()
        self.define_optimizer()

    def build_model(self):
        '''
            Builds the prediction node of the model.
        '''

        # Before beginning, reset the graph.
        tf.reset_default_graph()

        # define placeholders
        self.x = tf.placeholder(tf.float32, shape=(self.batch_size, self.timesteps, self.num_features), name='x')
        self.y = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_classes), name='y')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # normalize input
        normalized_input = tf.layers.batch_normalization(self.x, axis=2, training=self.is_training)
        
        # convert input tensor to list of tensors, for input in static_rnn() function.      
        inputs = tf.split(normalized_input, self.timesteps, axis=1) # convert (batch_size, timesteps, num_features) to list of (batch_size, 1, num_features)
        inputs = [tf.squeeze(inp) for inp in inputs] # convert (batch_size, 1, num_features) to (batch_size, num_features)

        # RNN layers
        rnn1_out = self._lstm_layer(inputs, 32, "lstm-1")
        rnn2_out = self._lstm_layer(rnn1_out, 64, "lstm-2")

        # Get last node
        final_node = rnn2_out[-1]

        # Dense layers
        dense1_out = self._dense_layer(final_node, 32, 'dense-1', activation=tf.nn.elu)
        dense2_out = self._dense_layer(dense1_out, self.num_classes, 'final', activation=tf.nn.softmax)

        self.prediction = dense2_out
        print(dense2_out.shape)

    def define_loss(self):
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.prediction)

    def define_optimizer(self):
        optimizer = tf.train.AdamOptimizer()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.loss)

    def train(self, path, epochs=50, log_step=10, resume=False):
        '''
            Trains the fall detection model on provided data.
            Args:
                path : Path to dataset directory
        '''

        x, y = self._read_data(path)
        data = Data(x, y)

        with tf.Session() as session:
            if resume:
                if os.path.isfile(self.model_file):
                    self.saver.restore(session, self.model_file)
                else:
                    print("No previous checkpoint file found, restarting training...")

            session.run(tf.global_variables_initializer())

            for e in range(epochs):
                avg_loss = 0.0
                for batch_x, batch_y in data.next_batch():
                    batch_loss, _ = session.run([self.loss, self.train_step], 
                                    feed_dict={self.x: x, self.y: y, self.is_training: True})
                    avg_loss += batch_loss

                if e%log_step == 0:
                    print("Average loss for epoch {} = {}.".format(e, avg_loss))
                    
                    # Save the model state
                    fname = self.saver.save(session, self.model_file)
                    print("Session saved in {}".format(fname))

        print("Training complete!")

    def _lstm_layer(self, inputs, num_units, scope, cell_type='LSTM'):
        '''
            Implements a LSTM layer.
            Args:
                inputs : A list of length `timesteps` containing tensors of shape (batch size, num_features).
                num_units : Number of units in RNN cell.
            Returns:
                List (of len `timesteps`) of outputs at each timestep, (batch size, num_units).
        '''

        with tf.variable_scope(scope):
            if cell_type == 'LSTM':
                lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)
            elif cell_type == 'GRU':
                lstm_cell = tf.nn.rnn_cell.GRUCell(num_units)
            else:
                raise NotImplementedError("{} type not implemented.".format(cell_type))

            outputs, states = tf.nn.static_rnn(lstm_cell, inputs, dtype=tf.float32)
        return outputs

    def _dense_layer(self, inputs, num_units, scope, activation=None):
        '''
            Implements a dense layer.
            Args:
                inputs : Tensor of shape (batch_size, x).
                num_units : Number of nodes in dense layer.
                scope : String - scope of layer
                activation : If not None, this activation function is applied to output.
            Returns:
                Tensor of shape (batch_size, num_units)
        '''

        with tf.variable_scope(scope):
            w = tf.get_variable("w", shape=(inputs.shape[-1].value, num_units))
            b = tf.get_variable("b", shape=(num_units), initializer=tf.constant_initializer(0.1))

            out = tf.matmul(inputs, w) + b
            if activation is not None:
                out = activation(out)

        return out

    def _read_data(self, path):
        '''
            Reads dataset from directory.
            Args:
                path : Path to dataset directory.
            Returns:
                (x, y) tuple where:
                    x : Numpy array of shape (Batch, timesteps, features)
                    y : Numpy array of shape (Batch, num_classes)
        '''
        assert os.path.isdir(path), "{} is not a valid directory.".format(path)

        raise NotImplementedError

if __name__ == '__main__':
    fallDetector = FallDetector(32, 9, 100, 13)