from experiment.qa.model.helper.pooling_helper import maxpool

from experiment.qa.model.lstm import BiLSTMModel


class LSTMLSTMModel(BiLSTMModel):
    def __init__(self, config, config_global, logger):
        super(LSTMLSTMModel, self).__init__(config, config_global, logger)

    def build(self, data, sess):
        self.build_input(data, sess)
        self.initialize_weights()

        lstm_representation_question_1 = self.bilstm_representation_raw(
            self.embeddings_question,
            self.input_question,
            re_use_lstm=False,
            name='lstm_1'
        )
        lstm_representation_answer_good_1 = self.bilstm_representation_raw(
            self.embeddings_answer_good,
            self.input_answer_good,
            re_use_lstm=True,
            name='lstm_1'
        )
        lstm_representation_answer_bad_1 = self.bilstm_representation_raw(
            self.embeddings_answer_bad,
            self.input_answer_bad,
            re_use_lstm=True,
            name='lstm_1'
        )

        lstm_representation_question_2 = self.bilstm_representation_raw(
            lstm_representation_question_1,
            self.input_question,
            re_use_lstm=False,
            name='lstm_2'
        )
        lstm_representation_answer_good_2 = self.bilstm_representation_raw(
            lstm_representation_answer_good_1,
            self.input_answer_good,
            re_use_lstm=True,
            name='lstm_2'
        )
        lstm_representation_answer_bad_2 = self.bilstm_representation_raw(
            lstm_representation_answer_bad_1,
            self.input_answer_bad,
            re_use_lstm=True,
            name='lstm_2'
        )

        pooled_representation_question = maxpool(lstm_representation_question_2)
        pooled_representation_answer_good = maxpool(lstm_representation_answer_good_2)
        pooled_representation_answer_bad = maxpool(lstm_representation_answer_bad_2)

        self.create_outputs(
            pooled_representation_question,
            pooled_representation_answer_good,
            pooled_representation_question,
            pooled_representation_answer_bad
        )


component = LSTMLSTMModel
