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
from sklearn.metrics import confusion_matrix

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
        - X_train, X_test : matrix of size (len(query_indices), number of features)
        - y_train, y_test : array of size len(query_indices), containing the label of each query point
        - label_names : dictionary linking the class index label and the label name

    Attributes:
        - X_train, X_test, y_train, y_test, label_names : (as input)

    Methods :
        - random_forest : return a trained classifier
        - evaluate(classifier)

    """

    def __init__(self, X_train, X_test, y_train, y_test, label_names):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.label_names = label_names

    def random_forest(self):
        clf = RandomForestClassifier(random_state=0)
        clf.fit(self.X_train, self.y_train)
        return clf

    def evaluate(self, clf, results_dir=None, filename=None):
        y_pred = clf.predict(self.X_test)
        score = np.sum(y_pred == self.y_test)/len(self.y_test)

        labels = list(self.label_names.keys())
        considered_labels = [l for l in labels if l in self.y_test]

        prop_M_confus = confusion_matrix(self.y_test, y_pred, labels=considered_labels, normalize="true")

        # class statistics
        recall = np.diag(prop_M_confus)
        precision = np.diag(confusion_matrix(self.y_test, y_pred, labels=considered_labels, normalize="pred"))
        F_measure = 2*recall*precision/(recall+precision)

        # general statistics
        R = np.mean(recall)
        P = np.mean(precision)
        F = 2*R*P/(R+P)

        # plot the confusion matrix (normalized according with the expected predictions (ie : ground truth))
        if (results_dir is not None) and (filename is not None):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(prop_M_confus, cmap='Reds', vmin=0, vmax=1)
            fig.colorbar(cax)

            names = [self.label_names[l] for l in considered_labels]
            ticks = np.arange(0,len(names),1)
            ax.set_xticks(ticks)
            plt.xticks(rotation=90)
            ax.set_yticks(ticks)

            ax.set_xticklabels(names)
            ax.set_yticklabels(names)

            plt.tight_layout()
            plt.savefig(results_dir+"/"+filename)

        measures = {
        "accuracy":score,
        "recall_by_class":recall,
        "precision_by_class":precision,
        "F_by_class":F_measure,
        "mean_recall":R,
        "mean_precision":P,
        "global_F":F,
        "considered_labels":considered_labels,
        }

        return y_pred, measures

    def get_classification_statistics(self, y_pred):
        """
        Get statistics about what class were the most misclassified.
        """
        misclassified = {}
        confused = {}
        indices_misclassified = np.flatnonzero(self.y_test != y_pred)
        for label in self.label_names.keys():
            # count how many points were misclassified in this class
            if sum(self.y_test==label)==0:
                misclassified[label] = 100.0
            else :
                misclassified[label] = np.round(1 - sum(self.y_test[indices_misclassified]==label)/sum(self.y_test==label),2)*100
            # find what is the class these points were the most classified as,
            # and the amount of this confusion
            lab_class = y_pred[np.flatnonzero(self.y_test==label)]
            lab_class = lab_class[lab_class != label]
            count_class = np.array([np.sum(lab_class==lab) for lab in self.label_names.keys()])
            confused_class = np.argmax(count_class)
            if len(lab_class)==0:
                confused[label] = ["", 0]
            else :
                confused[label] = [self.label_names[confused_class], np.round(count_class[confused_class]/len(lab_class),2)*100]

        d_name = max([len(self.label_names[label]) for label in self.label_names.keys()])+2
        f = "   - class {0:<%d} : {1:>5}%% correctly classified, else mainly confused with {2:>%d} (proportion : {3:>5}%%)" % (d_name, d_name)

        print("\nMisclassification statistics :")
        for label in self.label_names.keys():
            print(f.format(
                    "'"+self.label_names[label]+"'",
                    str(misclassified.get(label, 0))[:5],
                    "'"+str(confused.get(label,["",0])[0])+"'",
                    str(confused.get(label,["",0])[1])[:5]))
        print("")
