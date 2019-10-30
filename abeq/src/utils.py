import logging
import sys, os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER = logging.getLogger()
    FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")

    if not os.path.isdir('./logs'):
        os.makedirs('./logs')

    return LOGGER

class ModelFactory(object):
    def __init__(self, name, config, logger):
        logger.info('Selecting model => {0}'.format(name))

        if name == 'LogisticRegression':
            self.model = LogisticRegression(**config[name])
        elif name == 'RandomForest':
            self.model = RandomForestClassifier()
        # elif name == 'LightGBM':
        #     self.model = LGBMClassifier()
        else:
            logger.error('{0} is not implemented'.format(name))
            raise NotImplementedError()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        prediction = self.model.predict_proba(X)
        return prediction

    def predict_class(self, X):
        prediction = self.model.predict(X)
        return prediction