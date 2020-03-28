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
        - cloud
        - train_indices, test_indices : "query_indices" for both set
        - X_train, X_test : matrix of size (len(query_indices), number of features)

    Attributes:
        - cloud, train_indices, X_train, X_test : (as input)
        - y_train, y_test : labels for the query_indices

    Methods :
        - random_forest : return a trained classifier
        - evaluate(classifier)

    """

    def __init__(self, cloud, train_indices, test_indices, X_train, X_test):

        self.cloud = cloud
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = self.cloud.labels[self.train_indices]
        self.y_test = self.cloud.labels[self.test_indices]

    def random_forest(self):
        clf = RandomForestClassifier(random_state=0)
        clf.fit(self.X_train, self.y_train)
        return clf

    def evaluate(self, clf):
        y_pred = clf.predict(self.X_test)
        score = np.sum(y_pred == self.y_test)/len(self.y_test)
        return y_pred, score
