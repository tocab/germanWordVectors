import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
import sys


class prediction_model:

    def __init__(self, max_seq_len, embedding_size, batch_size, sess):
        self.sess = sess
        self.batch_size = batch_size

        self.x_ph = tf.placeholder(tf.float32, [None, max_seq_len, embedding_size])
        self.y_ph = tf.placeholder(tf.int32, [None, 1])

        cell_fw = LSTMCell(128)
        cell_bw = LSTMCell(128)

        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                 inputs=self.x_ph,
                                                                 dtype=tf.float32,
                                                                 time_major=False)

        outputs = tf.concat(outputs, 2)
        outputs = tf.keras.layers.GlobalMaxPool1D()(outputs)
        outputs = tf.layers.dense(inputs=outputs, units=3)

        self.probs = tf.nn.softmax(outputs)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.one_hot(self.y_ph, 3),
            logits=outputs
        )

        self.loss = tf.reduce_mean(cross_entropy)
        self.adam_opt = tf.train.AdamOptimizer().minimize(self.loss)

        sess.run(tf.global_variables_initializer())

    def train(self, x_train, y_train, x_val, y_val, epochs=3):
        y_val = y_val.reshape([-1, 1])
        y_train = y_train.reshape([-1, 1])
        i = 0
        start_batch = 0
        validation_losses = []
        while i < epochs:
            x_batch = x_train[start_batch:start_batch + self.batch_size, :, :]
            y_batch = y_train[start_batch:start_batch + self.batch_size, :]

            feed_dict = {
                self.x_ph: x_batch,
                self.y_ph: y_batch
            }

            loss, _ = self.sess.run([self.loss, self.adam_opt], feed_dict)

            val_loss = self.sess.run(self.loss, {self.x_ph: x_val, self.y_ph: y_val})
            validation_losses.append(val_loss)
            print("epoch:", i, " and train loss:", loss, "and val loss:", val_loss)
            start_batch += self.batch_size

            if start_batch >= x_train.shape[0]:
                start_batch = 0
                i += 1

    def eval_model(self, x_val, y_val, metric="accuracy"):
        probabilities = self.sess.run(self.probs, {self.x_ph: x_val})

        labels_ph = tf.placeholder(tf.int32)
        predictions_ph = tf.placeholder(tf.int32)

        if metric == "accuracy":
            print("Evaluating with accuracy")
            a_metric = tf.keras.metrics.binary_accuracy(y_true=labels_ph, y_pred=predictions_ph)
            self.sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        else:
            print("Evaluation metric not found.")
            sys.exit()

        for i in range(probabilities.shape[1]):
            score = self.sess.run(a_metric, {labels_ph: y_val == i, predictions_ph: np.argmax(probabilities, 1) == i})
            print("class", i, metric, "score:", score)