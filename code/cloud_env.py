#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31/03/2020

Projet NPM 3D

Jeffery Durand and Ginger Delmas

################################

Cloud environment file : here is defined everything that deals directly with the cloud :
    - class cloud
    - class train_cloud
    - class test_cloud
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


# *******************************CLASS SEPARATION***********************************************


class train_cloud(cloud):
    """
        Basic class for a training point cloud, deriving from the cloud class. Takes:
            - ply_path, save_dir, and save_file: see "cloud" class
            - load_if_possible : if True, loads the previously saved version if possible
            - num_points_per_label : number of points to randomly sample per labeled class

        Attributes :
            - samples_indices : dictionnary linking labels to indices (in the cloud) of the randomly sampled points
    """

    def __init__(self, ply_path, save_dir, save_file, load_if_possible=True,
                 num_points_per_label=500):

        # call the "cloud" class __init__()
        super().__init__(ply_path, save_dir, save_file)

        # if load() succeeds, skip initializing
        if not self.load(load_if_possible):
            # include labels since this is the train set
            self.fetch_points(include_labels=True)
            self.sample_n_points_per_label(num_points_per_label)


    def sample_n_points_per_label(self, num_points_per_label):
        """
            Sample a fixed number of points per labeled class to:
                1) limit computation time
                2) avoid bias more common classes
        """

        # dictionary of sampled points
        self.samples_indices = {}

        for label in self.label_names.keys():

            # skip 'Unclassified' points
            if self.label_names[label] == 'Unclassified': continue

            # randomly sample points corresponding to the given label
            # choose as many as possible if there aren't enough points
            label_indices = np.flatnonzero(self.labels == label)
            try:
                sampled_indices = np.random.choice(label_indices, num_points_per_label, replace=False)
            except ValueError:
                sampled_indices = label_indices
                print("Warning: class '{}' only has {}/{} points".format(
                    self.label_names[label], len(label_indices), num_points_per_label))

            # add to the recorded sampled points
            self.samples_indices[label] = sampled_indices

        return self.samples_indices


# *******************************CLASS SEPARATION***********************************************


class test_cloud(cloud):
    """ Basic class for a test point cloud """

    def __init__(self, ply_path, save_dir, save_file, load_if_possible=True):

        # call the "cloud" class __init__()
        super().__init__(ply_path, save_dir, save_file)

        # if load_if_possible is True and load() succeeds, skip initializing
        if not self.load(load_if_possible):
            # skip labels since this is the test set
            self.fetch_points(include_labels=False)
