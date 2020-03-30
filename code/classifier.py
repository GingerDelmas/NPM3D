#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31/03/2020

Projet NPM 3D

Jeffery Durand and Ginger Delmas

################################

Classifier file : here is defined everything to classify points :
    - class classifier
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
        - cloud_tr : cloud of the train set
        - cloud_te : cloud of the test set, if different from the training cloud
        - test_cloud_diff : tells whether to use the input cloud_te
        - train_indices, test_indices : "query_indices" for both set
        - X_train, X_test : matrix of size (len(query_indices), number of features)

    Attributes:
        - cloud_tr, cloud_te, train_indices, test_indices, X_train, X_test : (as input)
        - y_train, y_test : labels for the query_indices

    Methods :
        - random_forest : return a trained classifier
        - evaluate(classifier)

    """

    def __init__(self, cloud_tr, train_indices, test_indices, X_train, X_test, test_cloud_diff=False, cloud_te=None):

        self.cloud_tr = cloud_tr
        if test_cloud_diff :
            self.cloud_te = cloud_te
        else :
            self.cloud_te = cloud_tr
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = self.cloud_tr.labels[self.train_indices]
        self.y_test = self.cloud_te.labels[self.test_indices]

    def random_forest(self):
        clf = RandomForestClassifier(random_state=0)
        clf.fit(self.X_train, self.y_train)
        return clf

    def evaluate(self, clf):
        y_pred = clf.predict(self.X_test)
        score = np.sum(y_pred == self.y_test)/len(self.y_test)
        return y_pred, score

    def get_classification_statistics(self, y_pred):
        """
        Get statistics about what class were the most misclassified.
        """
        misclassified = {}
        confused = {}
        indices_misclassified = np.flatnonzero(self.y_test != y_pred)
        for label in self.cloud_te.label_names.keys():
            # count how many points were misclassified in this class
            if sum(self.y_test==label)==0:
                misclassified[label] = 100.0
            else :
                misclassified[label] = np.round(1 - sum(self.y_test[indices_misclassified]==label)/sum(self.y_test==label),2)*100
            # find what is the class these points were the most classified as,
            # and the amount of this confusion
            lab_class = y_pred[np.flatnonzero(self.y_test==label)]
            lab_class = lab_class[lab_class != label]
            count_class = np.array([np.sum(lab_class==lab) for lab in self.cloud_te.label_names.keys()])
            confused_class = np.argmax(count_class)
            if len(lab_class)==0:
                confused[label] = ["", 0]
            else :
                confused[label] = [self.cloud_te.label_names[confused_class], np.round(count_class[confused_class]/len(lab_class),2)*100]

        d_name = max([len(self.cloud_te.label_names[label]) for label in self.cloud_te.label_names.keys()])+2
        #d1 = len(str(max([misclassified.get(label, 0) for label in self.label_names.keys()])))
        d_pts = len(str(max([len(self.cloud_te.test_samples_indices.get(label, [])) for label in self.cloud_te.label_names.keys()])))

        f = "   - class {0:<%d} : {1:>5}%% correctly classified, else mainly confused with {2:>%d} (proportion : {3:>5}%%) [{4:>%d} points]" % (d_name, d_name, d_pts)

        print("\nMisclassification statistics :")
        for label in self.cloud_te.label_names.keys():
            print(f.format(
                    "'"+self.cloud_te.label_names[label]+"'",
                    str(misclassified.get(label, 0))[:5],
                    "'"+str(confused.get(label,["",0])[0])+"'",
                    str(confused.get(label,["",0])[1])[:5],
                    len(self.cloud_te.test_samples_indices.get(label, []))))
        print("")
