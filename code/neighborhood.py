#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31/03/2020

Projet NPM 3D

Jeffery Durand and Ginger Delmas

################################

Neighborhood file : here is defined everything to compute neighborhood size (and linked values) :
    - class neighborhood
    - class neighborhood_finder
"""

################################################################################
# IMPORTS
################################################################################

import numpy as np
from sklearn.neighbors import KDTree

from utils import *
from cloud_env import *

################################################################################
# CLASS DEFINITIONS
################################################################################

class neighborhood:
    """
        Container for a point's neighborhood.

        Note that since the normal and eigenvalues (of the structure tensor)
        were calculated to find the neighborhood, these are also stored.
    """

    def __init(self, cloud, indices, normal, eigenvalues):
        self.indices = indices
        self.cloud = cloud
        self.k = len(indices)
        self.normal = normal
        self.eigenvalues = eigenvalues

    def points(self):
        return self.cloud.points[self.indices]


# *******************************CLASS SEPARATION***********************************************


class neighborhood_finder(saveable):
    """
        Find the neighborhood size of each point denoted by query_indices in a cloud.
        Use case : query_indices represent the set of points sampled from the cloud
        on which computing the features and all.

        In:
            - cloud : the relevant 'cloud' class
            - query_indices : indices of the points in the cloud of which we want neighborhoods
            - save_dir and save_file : see "saveable" class
            - load_if_possible : if True, loads the previously saved version if possible
            - k_min, k_max : values of k to study as potential neighborhood sizes for each point

        The different optimal neighbourhood finder methods which are:
            - k_dummy()
            - k_critical_curvature()
            - k_min_shannon_entropy()
            - k_min_eigenentopy()
        should return:

        Out:
            - neighborhoods_size : array of size len(query_indices), containing the best k value for each query points.
                (dtype : uint8, to save memory (this assume a point does not have more than 255 neighbors))
            - eigenvalues : shape (len(query_indices), 3)
                            contains l1, l2, l3 for every point of query_indices
                            and the right value of "k" (NOTE : the eigenvalues should NOT be normalized !)
            - normals : shape (len(query_indices), 3)
                        contains normals coordinates for every point of query_indices
                        and the right value of "k"

        Attributes:
            - cloud (input)
            - query_indices (input)
            - k_min, k_max (input)
            - eigenvalues_tmp : shape (len(query_indices), k_max - k_min + 1, 3)
                                contains l1, l2, l3 for every point of query_indices
                                and k between k_min and k_max
            - normals_tmp : shape (len(query_indices), k_max - k_min + 1, 3),
                            contains the normal coordinates for every point of query_indices
                            and k between k_min and k_max
    """

    def __init__(self, cloud, query_indices, save_dir, save_file, load_if_possible=True, k_min=10, k_max=100):

        # call the "saveable" class __init__()
        super().__init__(save_dir, save_file)

        # store arguments
        self.cloud = cloud
        self.query_indices = query_indices
        self.k_min = k_min
        self.k_max = k_max

        # if load() succeeds, skip initializing
        if not self.load(load_if_possible):
            # calculate all needed data
            self.eigenvalues_tmp, self.normals_tmp = self.compute_over_k_range()


    def compute_over_k_range(self):
        """
            Loop through all considered k values and store needed data:
                - eigenvectors of the structure tensor l1, l2, l3
                - the normal vector found with this neighborhood

            Vectorized by querying the knn for all query_indices at a given k

            Out: both outputs have shape (len(query_indices), k_max - k_min + 1, 3)
                - eigenvalues : np array storing l1, l2, l3 for every value of query_indices and k,
                - normals : np array storing the normal vector found for every value of query_indices and k
        """

        # empty containers for the output
        eigenvalues = np.empty((len(self.query_indices), self.k_max - self.k_min + 1, 3))
        normals = np.empty((len(self.query_indices), self.k_max - self.k_min + 1, 3))

        for k in range(self.k_min, self.k_max+1): # this includes calculus for k = k_max
            knns = self.cloud.tree.query(self.cloud.points[self.query_indices], k, return_distance=False)
            eigenvalues[:,k-self.k_min,:], normals[:,k-self.k_min,:] = self.local_PCA(knns)

        return eigenvalues, normals


    def local_PCA(self, knns):
        """
            Given knns, find the normals and eigenvalues of the 3D structure tensor.

            Out: both outputs have shape (len(query_indices), 3)
                - eigenvalues : np array storing l1, l2, l3 for every value of query_indices,
                - normals : np array storing the normal vector found for every value of query_indices
        """
        # empty containers for the output
        eigenvalues = np.zeros((len(self.query_indices), 3))
        normals = np.zeros((len(self.query_indices), 3))

        # define useful function here
        vec2mat = lambda v : np.outer(v,v)

        for q in range(len(self.query_indices)):

            pts = self.cloud.points[knns[q]]

            centroid = np.mean(pts, axis=0)

            # Compute the covariance matrix
            cov = np.apply_along_axis(vec2mat, 1, pts - centroid)
            cov = sum(cov)/len(cov)

            # Compute the eigenvalues and eigenvectors
            eigenvalues[q], eigenvector = np.linalg.eigh(cov)
            normals[q] = eigenvector[:,0]

        return eigenvalues, normals

    def k_dummy(self):
        """
        Returns an array full of a unic k value.
        """
        neighborhoods_size = np.ones(len(self.query_indices), dtype="uint8")*self.k_min
        eigenvalues = self.eigenvalues_tmp[:,0,:]
        normals = self.normals_tmp[:,0,:]

        return neighborhoods_size, eigenvalues, normals

    def k_critical_curvature(self): # TODO
        """
            k maximizing the change in curvature C = l3 / (l1 + l2 + l3)
            where li is the ith biggest eigenvalue of the structure tensor
        """

        # TO IMPLEMENT
        neighborhoods_size = np.ones(len(query_points), dtype="uint8")
        eigenvalues = np.zeros((len(self.query_indices), 3))
        normals = np.zeros((len(self.query_indices), 3))

        return neighborhoods_size, eigenvalues, normals


    def k_min_shannon_entropy(self): # TODO
        """
            k minimizing the entropy Edim = - L*ln(L) - P*ln(P) - S*ln(S)
            where L, P and S are the linearity, planarity and sphericity
        """

        # TO IMPLEMENT
        neighborhoods_size = np.ones(len(query_points), dtype="uint8")
        eigenvalues = np.zeros((len(self.query_indices), 3))
        normals = np.zeros((len(self.query_indices), 3))

        return neighborhoods_size, eigenvalues, normals


    def k_min_eigenentopy(self): # TODO
        """
            k minimizing the entropy El = - e1*ln(e1) - e2*ln(e2) - e2*ln(e2)
            where ei is the normalized ith biggest eigenvalue of the structure tensor
        """

        # TO IMPLEMENT
        neighborhoods_size = np.ones(len(query_points), dtype="uint8")
        eigenvalues = np.zeros((len(self.query_indices), 3))
        normals = np.zeros((len(self.query_indices), 3))

        return neighborhoods_size, eigenvalues, normals
