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

# Documentation update: 09/01/2020

from ..settings import *
from ..utils import *

from .dataset import *
from .data_processor import *


class Feature_Selector():
    def __init__(self,type,parameters, input_features, input_labels):
        self.type=type
        self.parameters=parameters
        self.input_features=input_features
        self.input_labels=input_labels
        self.feature_selector=create_selector()


    def create_selector(self):
        if self.type=='variance':
            selector=feature_selection.VarianceThreshold(**self.parameters)
        elif self.type=='kbest':
            selector=feature_selection.SelectKBest(feature_selection.chi2, **self.parameters)
        elif self.type=='fpr':
            selector=feature_selection.SelectFpr(feature_selection.chi2, **self.parameters)
        return selector
