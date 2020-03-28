#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31/03/2020

Projet NPM 3D

Jeffery Durand and Ginger Delmas

################################

Cloud environment file : here is defined everything that deals directly with the cloud :
    - class cloud
    - class train_test_cloud
"""

################################################################################
# IMPORTS
################################################################################

import numpy as np
from sklearn.neighbors import KDTree

from utils import *

################################################################################
# GLOBAL VARIABLES
################################################################################

# it should be "class", but on mini data files
name_of_class_label = "scalar_class"

################################################################################
# CLASS DEFINITIONS
################################################################################

class cloud(saveable):
    """
        Basic class for a point cloud. Takes:
            - ply_path : the path to the .ply file containing the cloud
            - save_dir and save_file : see "saveable" class
            - include_labels : boolean indicating if we should (or can) take labels

        Attributes:
            - label_names : [hard coded] name of the different admitted labels
            - ply_path : (as input)
            - points : matrix of size (number of points , 3) containing the coordinates of each point
            - labels : array of size (number of points), containing the labels for each point
            - tree : KDTree based on "points"

    """

    def __init__(self, ply_path, save_dir, save_file):

        # call the "saveable" class __init__()
        super().__init__(save_dir, save_file)

        # the set categories of the data
        self.label_names = {0: 'Unclassified',
                            1: 'Ground',
                            2: 'Building',
                            3: 'Poles',
                            4: 'Pedestrians',
                            5: 'Cars',
                            6: 'Vegetation'}

        # save path to ply file
        self.ply_path = ply_path


    def fetch_points(self, include_labels: bool):
        # read the ply file and store content
        cloud_ply = read_ply(self.ply_path)
        self.points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        if include_labels: self.labels = cloud_ply[name_of_class_label]
        # make the KD Tree
        self.tree = KDTree(self.points)

    def get_statistics(self):
        """
        Get statistics about the number of element per class.
        This won't work if the attribute "labels" doesn't exist.
        """

        print("\nStatistics :")
        try:
            for label in self.label_names.keys():
                print("   - class '{}' represent {} points.".format(
                        self.label_names[label],
                        len(np.flatnonzero(self.labels == label))))
        except:
            print("Attribute 'labels' does not exist : please include labels on load.")
        print("\n")

# *******************************CLASS SEPARATION***********************************************


class train_test_cloud(cloud):
    """
        Basic class for a training point cloud, deriving from the cloud class. Takes:
            - ply_path, save_dir, and save_file: see "cloud" class
            - load_if_possible : if True, loads the previously saved version if possible
            - num_points_per_label_train : number of points to randomly sample per labeled class
                                           for the train set
            - num_points_per_label_test : idem, but for the test set

        Attributes :
            - train_samples_indices : dictionary linking labels to indices (in the cloud)
                                        of the randomly sampled points for the train set
            - test_samples_indices : as train_samples_indices, but for the test set

    """

    def __init__(self, ply_path, save_dir, save_file, load_if_possible=True,
                 num_points_per_label_train=500, num_points_per_label_test=500):

        # call the "cloud" class __init__()
        super().__init__(ply_path, save_dir, save_file)

        # if load() succeeds, skip initializing
        if not self.load(load_if_possible):
            # include labels since this is the train set
            self.fetch_points(include_labels=True)
            self.train_samples_indices = self.sample_n_points_per_label(num_points_per_label_train, train=True)
            self.test_samples_indices = self.sample_n_points_per_label(num_points_per_label_test, test=True)


    def sample_n_points_per_label(self, num_points_per_label, train=False, test=False):
        """
            Sample a fixed number of points per labeled class to:
                1) limit computation time
                2) avoid bias more common classes

            In :
                - num_points_per_label : number of points to sample
                - train : whether the sampling is performed for the train set.
                - test : whether the sampling is for the test set
                         If True, points used for the train set are not samplable.

            If both train and test are set to False, then this just samples some points.

            Out :
                - samples_indices : dictionary linking labels to indices (sampled points from the cloud)
        """

        # dictionary of sampled points
        samples_indices = {}

        for label in self.label_names.keys():

            # skip 'Unclassified' points
            if self.label_names[label] == 'Unclassified': continue

            # randomly sample points corresponding to the given label
            # choose as many as possible if there aren't enough points

            label_indices = np.flatnonzero(self.labels == label)

            if test : # don't consider points that were already sampled for the train set
                label_indices = set(label_indices).difference(set(self.train_samples_indices[label]))
                label_indices = np.array(list(label_indices))

            try:
                sampled_indices = np.random.choice(label_indices, num_points_per_label, replace=False)
            except ValueError:
                sampled_indices = label_indices
                print("Warning: class '{}' only has {}/{} points (left)".format(
                    self.label_names[label], len(label_indices), num_points_per_label))

            samples_indices[label] = sampled_indices

        return samples_indices
