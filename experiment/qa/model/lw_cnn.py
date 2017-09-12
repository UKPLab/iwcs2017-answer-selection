from experiment.qa.model.cnn import CNNModel
from experiment.qa.model.helper.lw import LW
from experiment.qa.model.helper.pooling_helper import weighted_pooling


class LWCNNModel(CNNModel, LW):
    def __init__(self, config, config_global, logger):
        super(LWCNNModel, self).__init__(config, config_global, logger)
        self.shared_lw = self.config.get('shared_lw', False)

    def build(self, data, sess):
        self.build_input(data, sess)
        self.initialize_weights()

        raw_representation_question = self.cnn_representation_raw(self.embeddings_question, self.question_length)
        raw_representation_answer_good = self.cnn_representation_raw(self.embeddings_answer_good, self.answer_length)
        raw_representation_answer_bad = self.cnn_representation_raw(self.embeddings_answer_bad, self.answer_length)

        self.question_pooling_weight = self.positional_weighting(
            raw_representation_question,
            self.input_question,
            item_type='question' if not self.shared_lw else 'shared'
        )
        self.answer_good_pooling_weight = self.positional_weighting(
            raw_representation_answer_good,
            self.input_answer_good,
            item_type='answer' if not self.shared_lw else 'shared'
        )
        self.answer_bad_pooling_weight = self.positional_weighting(
            raw_representation_answer_bad,
            self.input_answer_bad,
            item_type='answer' if not self.shared_lw else 'shared'
        )

        pooled_representation_question = weighted_pooling(
            raw_representation_question, self.question_pooling_weight, self.input_question
        )
        pooled_representation_answer_good = weighted_pooling(
            raw_representation_answer_good, self.answer_good_pooling_weight, self.input_answer_good
        )
        pooled_representation_answer_bad = weighted_pooling(
            raw_representation_answer_bad, self.answer_bad_pooling_weight, self.input_answer_bad
        )

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
        LW.initialize_weights(self)


component = LWCNNModel
