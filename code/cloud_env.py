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
from tqdm import tqdm

from utils import *
from ply import *

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
            - cloud_path : the path to the .ply or .txt file containing the cloud
            - label_path : the path to the .labels file containing the labels, if any
            - save_dir and save_file : see "saveable" class
            - include_labels : boolean indicating if we should (or can) take labels

        Attributes:
            - label_names : [hard coded] name of the different admitted labels
            - cloud_path : (as input)
            - label_path : (as input)
            - points : matrix of size (number of points , 3) containing the coordinates of each point
            - labels : array of size (number of points), containing the labels for each point
            - tree : KDTree based on "points"

        Methods :
            - fetch_points
            - get_statistics

    """

    def __init__(self, cloud_path, save_dir, save_file=None, label_path=None):

        # call the "saveable" class __init__()
        identifiers = cloud_path.split('/')[-1]  # the filename
        super().__init__(save_dir, identifiers, save_file=save_file)

        # the set categories of the data
        self.label_names = {0: 'Unclassified'}

        # save path to ply file
        self.cloud_path = cloud_path
        self.label_path = label_path


    def fetch_points(self, include_labels: bool, file_type="ply"):
        if file_type=="ply":
            self.label_names = {0: 'Unclassified',
                                1: 'Ground',
                                2: 'Building',
                                3: 'Poles',
                                4: 'Pedestrians',
                                5: 'Cars',
                                6: 'Vegetation'}

            # read the ply file and store content
            cloud_ply = read_ply(self.cloud_path)
            self.points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
            if include_labels: 
                try:
                    self.labels = cloud_ply["class"]
                except ValueError:
                    self.labels = cloud_ply["scalar_class"]

        elif file_type=="txt":
            self.label_names = {0: 'Unclassified',
                                1: 'Ground',
                                2: 'Hard_scape',
                                3: 'Vegetation',
                                4: 'Cars',
                                5: 'Buildings'}

            if include_labels:
                # load labels
                f = open(self.label_path, "r")
                labels = f.read().splitlines()
                f.close()
                self.labels = np.array(labels).astype(int)

                # aggregate classes
                self.labels[self.labels==7] = 0 # scanning_artefacts -> Unclassified
                self.labels[self.labels==4] = 3 # high_vegetation, low_vegetation -> vegetation
                self.labels[self.labels==2] = 1 # natural_terrain, man-made_terrain -> ground

                # arange correctly the class id
                self.labels[self.labels==6] = 2
                self.labels[self.labels==8] = 4

            # load points
            f = open(self.cloud_path, "r")
            content = f.read().splitlines()
            f.close()

            self.points = np.zeros((len(content),3))
            for i,p in tqdm(enumerate(content)):
                self.points[i] = np.array(p.split(" ")[:3]).astype("float")

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
        print("")

# *******************************CLASS SEPARATION***********************************************


class train_test_cloud(cloud):
    """
        Basic class for a training point cloud, deriving from the cloud class. Takes:
            - cloud_path, label_path, save_dir, and save_file: see "cloud" class
            - load_if_possible : if True, loads the previously saved version if possible
            - num_points_per_label_train : number of points to randomly sample per labeled class
                                           for the train set. If it is "-1", all points are sampled.
            - num_points_per_label_test : idem, but for the test set. If it is "-1",
                                          all remaining points after the train sampling are sampled.
        Attributes :
            - train_samples_indices : dictionary linking labels to indices (in the cloud)
                                        of the randomly sampled points for the train set
            - test_samples_indices : as train_samples_indices, but for the test set

        Methods :
            - sample_n_points_per_label
            - hand_sampled_points
            - get_split_statistics
    """

    def __init__(self, cloud_path, save_dir, save_file=None, file_type="ply", label_path=None, load_if_possible=True,
                 num_points_per_label_train=500, num_points_per_label_test=500):

        # call the "cloud" class __init__()
        super().__init__(cloud_path, save_dir, save_file, label_path)

        # if load() succeeds, skip initializing
        if not self.load(load_if_possible):
            # include labels since this is the train set
            self.fetch_points(include_labels=True, file_type=file_type)
            self.train_samples_indices = self.sample_n_points_per_label(num_points_per_label_train, train=True)
            self.test_samples_indices = self.sample_n_points_per_label(num_points_per_label_test, test=True)


    def sample_n_points_per_label(self, num_points_per_label, train=False, test=False):
        """
            Sample a fixed number of points per labeled class to:
                1) limit computation time
                2) avoid bias more common classes

            In :
                - num_points_per_label : number of points to sample per label. If value is "-1",
                                         all (remaining) points are sampled.
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
                label_indices = np.array(list(label_indices)).astype(int)
            if num_points_per_label==-1: # take all points
                sampled_indices = label_indices
            else : # sample
                try:
                    sampled_indices = np.random.choice(label_indices, num_points_per_label, replace=False)
                except ValueError:
                    sampled_indices = label_indices
                    print("Warning: class '{}' only has {}/{} points".format(
                        self.label_names[label], len(label_indices), num_points_per_label))

            samples_indices[label] = sampled_indices

        return samples_indices

    def hand_sampled_points(self, samples_indices):
        """
        Convert the content of the dictionary "samples_indices" to get the indices only,
        into an array.
        """

        return np.concatenate([samples_indices[label]
                                        for label in samples_indices.keys()])

    def get_split_statistics(self):
        """
        Get statistics about the number of element per class in both the training
        and testing set.
        """

        d0 = max([len(self.label_names[label]) for label in self.label_names.keys()])+2
        d1 = max([len(self.train_samples_indices.get(label, [])) for label in self.label_names.keys()])
        d2 = max([len(self.test_samples_indices.get(label, [])) for label in self.label_names.keys()])
        d3 = max([len(np.flatnonzero(self.labels == label)) for label in self.label_names.keys()])

        d1 = len(str(d1))
        d2 = len(str(d2))
        d3 = len(str(d3))

        f = "   - class {0:<%d} : {1:>%d} (training), {2:>%d} (testing), {3:>%d} (total)" % (d0, d1, d2, d3)

        print("\nSplit statistics :")
        for label in self.label_names.keys():
            print(f.format(
                    "'"+self.label_names[label]+"'",
                    len(self.train_samples_indices.get(label, [])),
                    len(self.test_samples_indices.get(label, [])),
                    len(np.flatnonzero(self.labels == label))))
        print("")
