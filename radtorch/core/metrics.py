# Copyright (C) 2020 RADTorch and Mohamed Elbanan, MD
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/

# Update: 8/30/2020

from ..settings import *
from ..utils import *


class Metrics():
    def __init__(self, model,
                **kwargs):
        self.model = model
        for k, v in kwargs.items():
            setattr(self, k, v)

        if not self.model.trained_model:
            log('Error! Provided model was not trained/fitted. Please check.')
            pass

    def average_cv_accuracy(self):
        if self.model.cv:
          return self.model.scores.mean()
        else:
          log('Error! Training was done without cross validation. Please use test_accuracy() instead.', gui=gui)

    def test_accuracy(self) :
        return self.model.trained_model.score(self.model.test_features, self.model.test_labels)
        return acc

    def confusion_matrix(self,title='Confusion Matrix',cmap=None,normalize=False,figure_size=(8,6)):
        pred_labels=self.model.trained_model.predict(self.model.test_features)
        true_labels=self.model.test_labels
        cm = metrics.confusion_matrix(true_labels, pred_labels)
        show_confusion_matrix(cm=cm,
                              target_names=self.model.classes,
                              title=title,
                              cmap=cmap,
                              normalize=normalize,
                              figure_size=figure_size
                              )

    def roc(self, **kw):
        show_roc([self.model], **kw)
