import tensorflow as tf

from experiment.qa.model import QAModel, bias_variable, weight_variable
from experiment.qa.model.helper.pooling_helper import maxpool_tanh


class CNNModel(QAModel):
    def __init__(self, config, config_global, logger):
        super(CNNModel, self).__init__(config, config_global, logger)
        self.n_filters = self.config['filters']
        self.window_size = self.config['filter_size']

    def build(self, data, sess):
        self.build_input(data, sess)

        # we initialize the weights of the representation layers globally so that they can be applied to both, questions
        # and (good/bad)answers. This is an important part, otherwise results would be much worse.
        self.initialize_weights()

        representation_question = maxpool_tanh(
            self.cnn_representation_raw(
                self.embeddings_question,
                self.question_length
            ),
            self.input_question
        )
        representation_answer_good = maxpool_tanh(
            self.cnn_representation_raw(
                self.embeddings_answer_good,
                self.answer_length
            ),
            self.input_answer_good
        )
        representation_answer_bad = maxpool_tanh(
            self.cnn_representation_raw(
                self.embeddings_answer_bad,
                self.answer_length
            ),
            self.input_answer_bad
        )

        self.create_outputs(
            representation_question,
            representation_answer_good,
            representation_question,
            representation_answer_bad
        )

    def initialize_weights(self):
        """Global initialization of weights for the representation layer

        """
        self.W_conv1 = weight_variable('W_conv', [self.window_size, self.embedding_size, 1, self.n_filters])
        self.b_conv1 = bias_variable('b_conv', [self.n_filters])

    def cnn_representation_raw(self, item, sequence_length):
        """Creates a representation graph which retrieves a text item (represented by its word embeddings) and returns
        a vector-representation

        :param item: the text item. Can be question or (good/bad) answer
        :param sequence_length: maximum length of the text item
        :return: representation tensor
        """
        # we need to add another dimension, because cnn works on 3d data only
        cnn_input = tf.expand_dims(item, -1)

        convoluted = tf.nn.bias_add(
            tf.nn.conv2d(
                cnn_input,
                self.W_conv1,
                strides=[1, 1, self.embedding_size, 1],
                padding='SAME'
            ),
            self.b_conv1
        )

        return tf.reshape(convoluted, [-1, sequence_length, self.n_filters])


component = CNNModel
