'''
@author: Ankit Bindal
'''

import os
import pickle
import tensorflow as tf
from helper import Data, DataLoader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TB_DIR = "for_tensorboard-3dense"
model_dir = "saved_model-3dense"

class FallDetector:
    def __init__(self, batch_size, num_features, timesteps, num_classes):
        self.batch_size = batch_size
        self.num_features = num_features
        self.timesteps = timesteps
        self.num_classes = num_classes

        # Model checkpoint file
        os.makedirs(model_dir, exist_ok=True)
        self.model_file = os.path.join(model_dir, "model.ckpt")

        print("Building the model...")
        self.build_model()
        self.define_loss()
        self.define_optimizer()

        self.saver = tf.train.Saver()
        self.tensorboard_op = tf.summary.merge_all()
        print("Model built!")

    def build_model(self):
        '''
            Builds the prediction node of the model.
        '''

        # Before beginning, reset the graph.
        tf.reset_default_graph()

        # define placeholders
        self.x = tf.placeholder(tf.float32, shape=(self.batch_size, self.timesteps, self.num_features), name='x')
        self.y = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_classes), name='y')
        
        # convert input tensor to list of tensors, for input in static_rnn() function.      
        inputs = tf.split(self.x, self.timesteps, axis=1) # convert (batch_size, timesteps, num_features) to list of (batch_size, 1, num_features)
        inputs = [tf.squeeze(inp) for inp in inputs] # convert (batch_size, 1, num_features) to (batch_size, num_features)

        # RNN layers
        rnn1_out = self._lstm_layer(inputs, 32, "lstm-1")
        rnn2_out = self._lstm_layer(rnn1_out, 64, "lstm-2")

        # Get last node
        final_node = rnn2_out[-1]

        # Dense layers
        dense1_out = self._dense_layer(final_node, 32, 'dense-1', activation=tf.nn.elu)
        dense2_out = self._dense_layer(dense1_out, 8, 'dense-2', activation=tf.nn.elu)
        self.prediction = self._dense_layer(dense2_out, self.num_classes, 'Prediction', activation=tf.nn.softmax)

        # For accuracy
        with tf.variable_scope('accuracy'):
            pred_classes = tf.argmax(self.prediction, axis=-1)
            actual_classes = tf.argmax(self.y, axis=-1)
            correct = tf.equal(pred_classes, actual_classes)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='mean_accuracy')
            tf.summary.scalar("Accuracy", self.accuracy) # for tensorboard

    def define_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.prediction))
        tf.summary.scalar("Cross-entropy-loss", self.loss) # for tensorboard

    def define_optimizer(self):
        optimizer = tf.train.AdamOptimizer(0.0001)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.loss)

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

    def train(self, path, epochs=50, log_step=10, resume=False):
        '''
            Trains the fall detection model on provided data.
            Args:
                path : Path to dataset directory
        '''

        print("Beginning the training process...")

        if os.path.isfile('loaded_dataset.pkl'):
            data = pickle.load(open('loaded_dataset.pkl', 'rb'))
        else:
            data = DataLoader(path)
            pickle.dump(data, open('loaded_dataset.pkl', 'wb'))

        # load min validation loss
        valid_file = os.path.join(model_dir, "min_valid_loss.txt")
        if os.path.isfile(valid_file):
            min_valid_loss = float(open(valid_file).read().strip())
        else:
            min_valid_loss = 1000000.0

        with tf.Session() as session:
            writer = tf.summary.FileWriter(TB_DIR, session.graph)

            resumed = False
            if resume:
                try:
                    self.saver.restore(session, self.model_file)
                    resumed = True
                except:
                    print("No previous checkpoint file found, restarting training...")

            if not resumed:
                session.run(tf.global_variables_initializer())
            
            for e in range(epochs):
                avg_loss = []
                for batch_x, batch_y in data.next_batch(self.batch_size, training=True):
                    batch_loss, _, tb_op = session.run([self.loss, self.train_step, self.tensorboard_op], 
                                    feed_dict={self.x: batch_x, self.y: batch_y})
                    avg_loss.append(batch_loss)

                print("Average Loss for epoch {} = {}.".format(e, sum(avg_loss)/len(avg_loss)))
                
                if e%log_step == 0:
                    # Run for validation set
                    avg_loss = []
                    avg_accuracy = []
                    for batch_x, batch_y in data.next_batch(self.batch_size, training=False, validation=True):
                        batch_loss, batch_acc = session.run([self.loss, self.accuracy], 
                                        feed_dict={self.x: batch_x, self.y: batch_y})
                        avg_loss.append(batch_loss)
                        avg_accuracy.append(batch_acc)

                    avg_accuracy = sum(avg_accuracy) / len(avg_accuracy)
                    avg_loss = sum(avg_loss) / len(avg_loss)

                    if avg_loss < min_valid_loss:
                        min_valid_loss = avg_loss
                        with open(valid_file, 'w') as f:
                            f.write(str(avg_loss))
                        self.save_session(session)

                    print("Validation Error: {}. Validation Accuracy: {}".format(avg_loss, avg_accuracy))
                
                writer.add_summary(tb_op, e)
            
            writer.close()

        print("Training complete!")

        print(self.evaluate(data))

    def evaluate(self, data):
        with tf.Session() as session:
            self.saver.restore(session, self.model_file)

            avg_loss = []
            avg_accuracy = []
            for batch_x, batch_y in data.next_batch(batch_size=16, training=False):
                pred, loss_, acc_ = session.run([self.prediction, self.loss, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y})
                avg_loss.append(loss_)
                avg_accuracy.append(acc_)

            avg_loss = sum(avg_loss) / len(avg_loss)
            avg_accuracy = sum(avg_accuracy) / len(avg_accuracy)
            print("Average loss on evaluation set: {}".format(avg_loss))
            print("Average accuracy of evaluation set: {}".format(avg_accuracy))

        return avg_loss, avg_accuracy

    def save_session(self, sess):
        fname = self.saver.save(sess, self.model_file)
        print("Session saved in {}".format(fname))

if __name__ == '__main__':
    path = "MobiFall_Dataset_v2.0"
    # params : batch_size, num_features, timesteps, num_classes
    fallDetector = FallDetector(16, 9, 450, 2)
    fallDetector.train(path, epochs=100)

    # data = DataLoader(path)
    # fallDetector.evaluate(data)