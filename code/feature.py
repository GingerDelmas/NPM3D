#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31/03/2020

Projet NPM 3D

Jeffery Durand and Ginger Delmas

################################

Feature file : here is defined everything to compute features (and linked values) :
    - class features_finder
"""

################################################################################
# IMPORTS
################################################################################

import numpy as np
from sklearn.neighbors import KDTree

from utils import *
from cloud_env import *
from neighborhood import *

################################################################################
# CLASS DEFINITIONS
################################################################################


class features_finder(saveable):
    """
        Given a cloud, query indices, and corresponding neighborhoods, find features.

        In:
            - cloud and query_indices : see 'neighborhood_finder'
            - neighborhoods_size : array of size len(query_indices), containing the best k value for each query points.
                (dtype : uint8, to save memory (this assume a point does not have more than 255 neighbors))
            - eigenvalues : shape (len(query_indices), 3)
                            contains l1, l2, l3 for every point of query_indices
                            and the right value of "k"
            - normals : shape (len(query_indices), 3)
                        contains normals coordinates for every point of query_indices
                        and the right value of "k"
            - save_dir and save_file : see "saveable" class

        Out: functions dedicated to a single feature return a 1D numpy array.
            Those returning a collection of n different features return a
            numpy array of shape (len(query_indices), n)

        Methods returning a collection of features are:
            - features_dim()
            - features_2D()
            - features_3D()
            - features_2D_bins()
    """

    def __init__(self, cloud, query_indices, neighborhoods_size, eigenvalues, normals, save_dir, save_file):
        # call the "saveable" class __init__()
        super().__init__(save_dir, save_file)

        self.cloud = cloud
        self.query_indices = query_indices
        self.neighborhoods_size = neighborhoods_size
        self.eigenvalues = eigenvalues
        self.normals = normals


    def features_dim(self):
        """
            Find the dimensionality features:
                - linearity
                - planarity
                - sphericity
        """
        eps = 10**(-5) # to avoid errors when eigenvalues = 0
        lbda1, lbda2, lbda3 = self.eigenvalues[:,2], self.eigenvalues[:,1], self.eigenvalues[:,0]

        linearity = 1 - lbda2 / (lbda1 + eps)
        planarity = (lbda2 - lbda3) / (lbda1 + eps)
        sphericity = lbda3 / (lbda1 + eps)

        return linearity, planarity, sphericity


    def features_2D(self):
        """
            Return local 2D features (except bin-related ones):
                - geometric 2D properties:
                    - 2D radius
                    - 2D local point density
                - 2D shape features:
                    - eigenvalue sum
                    - eigenvalue ratio
        """
        pass


    def features_3D(self):
        """
            Return all local 3D features (including dimensionality features):
                - geometric 3D properties:
                    - absolute height
                    - radius
                    - maximum height difference
                    - height standard deviation
                    - local point density
                    - verticality
                - dimensionality features:
                    - linearity
                    - planarity
                    - sphericity
                - other 3D shape features:
                    - omnivariance
                    - anisotropy
                    - eigenentropy
                    - eigenvalue sum
                    - change of curvature
        """
        pass


    def features_2D_bins(self, side_length):
        """
            Return 2D accumulation map features:
                - number of points in bin
                - maximum height difference in bin
                - height standard deviation in bin
        """
        pass
