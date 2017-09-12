import tensorflow as tf

import experiment


class InsuranceQANoTraining(experiment.Training):
    """This is a replacement component that skips the training process"""

    def __init__(self, config, config_global, logger):
        super(InsuranceQANoTraining, self).__init__(config, config_global, logger)

    def start(self, model, data, sess):
        self.logger.info('Initializing all variables')
        sess.run(tf.initialize_all_variables())
        self.logger.info("Skipping training")

    def remove_checkpoints(self):
        pass


component = InsuranceQANoTraining
