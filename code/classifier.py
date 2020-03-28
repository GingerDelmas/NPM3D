#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31/03/2020

Projet NPM 3D

Jeffery Durand and Ginger Delmas

################################

Classifier file : here is defined everything to classify points :

"""

################################################################################
# IMPORTS
################################################################################

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from utils import *
from cloud_env import *
from neighborhood import *
from feature import *

################################################################################
# CLASS DEFINITIONS
################################################################################

class classifier:
    """ Container to test different kind of classifiers.

    In:
        - cloud and query_indices : see 'neighborhood_finder'
        - X : matrix of size (len(query_indices), number of features)

    Attributes:
        - cloud, query_indices, X : (as input)
        - y : labels of query_indices

    Methods :
        - randomForest : return a trained classifier
        - evaluate(classifier)

    """

    def __init__(self, cloud, query_indices, X):

        self.cloud = cloud
        self.query_indices = query_indices
        self.X = features
        self.y = self.cloud.labels[self.query_indices]

    def random_forest():
        clf = RandomForestClassifier(random_state=0)
        clf.fit(self.X, self.y)
        return clf

    def evaluate(clf):

        y_pred
