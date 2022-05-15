
from argparse import Namespace
from .modules import Conv2dClassifier
from ...base_lightning_modules.base_classification_model import BaseClassificationModel

class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)

        self.generator = Conv2dClassifier()
