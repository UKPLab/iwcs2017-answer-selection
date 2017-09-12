import tensorflow as tf
from experiment.qa.model.cnn import CNNModel
from experiment.qa.model.helper.pooling_helper import maxpool

from experiment.qa.model.lstm import BiLSTMModel


class LSTMCNNModel(CNNModel, BiLSTMModel):
    def __init__(self, config, config_global, logger):
        super(LSTMCNNModel, self).__init__(config, config_global, logger)

    def build(self, data, sess):
        self.build_input(data, sess)
        self.initialize_weights()

        raw_representation_question = self.cnn_representation_raw(self.embeddings_question, self.question_length)
        raw_representation_answer_good = self.cnn_representation_raw(self.embeddings_answer_good, self.answer_length)
        raw_representation_answer_bad = self.cnn_representation_raw(self.embeddings_answer_bad, self.answer_length)

        lstm_representation_question = self.bilstm_representation_raw(
            tf.nn.tanh(raw_representation_question),
            self.input_question,
            False
        )
        lstm_representation_answer_good = self.bilstm_representation_raw(
            tf.nn.tanh(raw_representation_answer_good),
            self.input_answer_good,
            True
        )
        lstm_representation_answer_bad = self.bilstm_representation_raw(
            tf.nn.tanh(raw_representation_answer_bad),
            self.input_answer_bad,
            True
        )

        pooled_representation_question = maxpool(lstm_representation_question)
        pooled_representation_answer_good = maxpool(lstm_representation_answer_good)
        pooled_representation_answer_bad = maxpool(lstm_representation_answer_bad)

        self.create_outputs(
            pooled_representation_question,
            pooled_representation_answer_good,
            pooled_representation_question,
            pooled_representation_answer_bad
        )

    def initialize_weights(self):
        """Global initialization of weights for the representation layer

        """
        CNNModel.initialize_weights(self)
        BiLSTMModel.initialize_weights(self)


component = LSTMCNNModel
