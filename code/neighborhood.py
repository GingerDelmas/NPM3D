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
from tqdm import tqdm
from sklearn.neighbors import KDTree
import time

from utils import *
from cloud_env import *

################################################################################
# GLOBAL VARIABLES
################################################################################
eps = 10**(-8) #  to avoid errors when eigenvalues = 0 (denominator, log)

################################################################################
# CLASS DEFINITIONS
################################################################################

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
            - eigs_to_test : shape (len(query_indices), k_max - k_min + 1, 3)
                                contains l1, l2, l3 for every point of query_indices
                                and k between k_min and k_max
            - normals_all : shape (len(query_indices), k_max - k_min + 1, 3),
                            contains the normal coordinates for every point of query_indices
                            and k between k_min and k_max
    """

    def __init__(self, cloud, query_indices, save_dir, save_file=None, load_if_possible=True, k_min=10, k_max=100):

        # call the "saveable" class __init__()
        identifiers = [cloud.points, query_indices]
        super().__init__(save_dir, identifiers, save_file=save_file)

        # store arguments
        self.cloud = cloud
        self.query_indices = query_indices

        # if load() does not succeed, calculate all needed data
        if not self.load(load_if_possible):
            self.k_min = self.k_min_all = k_min
            self.k_max = self.k_max_all = k_max
            self.eigs_all, self.normals_all, self.compute_time = self.compute_over_k_range(k_min, k_max)
            self.eigs_to_test, self.normals_to_test = self.eigs_all, self.normals_all
        # otherwise some data was already calculated, find what is missing
        else:
            # we are looking for lower k values then before
            if k_min < self.k_min_all:
                eigs_tmp, normals_tmp, compute_time_tmp = self.compute_over_k_range(k_min, self.k_min_all - 1)
                self.eigs_all = np.concatenate((eigs_tmp, self.eigs_all), axis=1)
                self.normals_all = np.concatenate((normals_tmp, self.normals_all), axis=1)
                self.compute_time += compute_time_tmp
                self.k_min_all = k_min
            # we are looking for higher k values then before
            if k_max > self.k_max_all:
                eigs_tmp, normals_tmp, compute_time_tmp = self.compute_over_k_range(self.k_max_all + 1, k_max)
                self.eigs_all = np.concatenate((self.eigs_all, eigs_tmp), axis=1)
                self.normals_all = np.concatenate((self.normals_all, normals_tmp), axis=1)
                self.compute_time += compute_time_tmp
                self.k_max_all = k_max
            # adjust the window considered to the current
            self.k_min = k_min; self.k_max = k_max;
            low_bound = k_min - self.k_min_all
            size = k_max - k_min + 1
            self.eigs_to_test = self.eigs_all[:,low_bound:low_bound + size]
            self.normals_to_test = self.normals_all[:,low_bound:low_bound + size]


    def compute_over_k_range(self, k_min, k_max):
        """
            Loop through all considered k values and store needed data:
                - eigenvectors of the structure tensor l1, l2, l3
                - the normal vector found with this neighborhood

            Vectorized by querying the knn for all query_indices at a given k

            Out: both outputs have shape (len(query_indices), k_max - k_min + 1, 3)
                - eigenvalues : np array storing l1, l2, l3 for every value of query_indices and k,
                - normals : np array storing the normal vector found for every value of query_indices and k
        """

        t0 = time.time()

        # empty containers for the output
        eigenvalues = np.empty((len(self.query_indices), k_max - k_min + 1, 3))
        normals = np.empty((len(self.query_indices), k_max - k_min + 1, 3))

        print('Trying k values from {} to {}. Completed:'.format(k_min, k_max))
        for k in tqdm(range(k_min, k_max + 1)): # this includes calculus for k = k_max
            knns = self.cloud.tree.query(self.cloud.points[self.query_indices], k, return_distance=False)
            eigenvalues[:,k-k_min,:], normals[:,k-k_min,:] = self.local_PCA(knns)
            # print('k = {}'.format(k))

        t1 = time.time()
        compute_time = t1 - t0

        return eigenvalues, normals, compute_time


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
        Returns an array full of a unique k value.
        """
        neighborhoods_size = np.ones(len(self.query_indices), dtype="uint8")*self.k_min
        eigenvalues = self.eigs_to_test[:,0,:]
        normals = self.normals_to_test[:,0,:]

        return neighborhoods_size, eigenvalues, normals

    def k_critical_curvature(self):
        """
            k maximizing the change in curvature C = l3 / (l1 + l2 + l3)
            where li is the ith biggest eigenvalue of the structure tensor
        """
        # find the best k for each query
        curvatures = self.eigs_to_test[...,2] / (np.sum(self.eigs_to_test, axis=2) + eps)
        bestk = np.argmax(curvatures, axis=1)

        # deduce the outputs
        neighborhoods_size = (bestk + self.k_min).astype("uint8")
        eigenvalues = self.eigs_to_test[range(len(bestk)),bestk,:]
        normals = self.normals_to_test[range(len(bestk)),bestk,:]

        return neighborhoods_size, eigenvalues, normals


    def k_min_shannon_entropy(self):
        """
            k minimizing the entropy Edim = - L*ln(L) - P*ln(P) - S*ln(S)
            where L, P and S are the linearity, planarity and sphericity
        """
        # find the best k for each query
        L = (self.eigs_to_test[...,0] - self.eigs_to_test[...,1]) / (self.eigs_to_test[...,0] + eps)
        P = (self.eigs_to_test[...,1] - self.eigs_to_test[...,2]) / (self.eigs_to_test[...,0] + eps)
        S = self.eigs_to_test[...,2] / (self.eigs_to_test[...,0] + eps)
        entropy = - L * np.log(L+eps) - P * np.log(P+eps) - S * np.log(S+eps)
        bestk = np.argmin(entropy, axis=1)

        # deduce the outputs
        neighborhoods_size = (bestk + self.k_min).astype("uint8")
        eigenvalues = self.eigs_to_test[range(len(bestk)),bestk,:]
        normals = self.normals_to_test[range(len(bestk)),bestk,:]

        return neighborhoods_size, eigenvalues, normals


    def k_min_eigenentropy(self):
        """
            k minimizing the entropy El = - e1*ln(e1) - e2*ln(e2) - e2*ln(e2)
            where ei is the normalized ith biggest eigenvalue of the structure tensor
        """

        # find the best k for each query
        entropy =  - np.sum(self.eigs_to_test * np.log(self.eigs_to_test + eps), axis=2)
        bestk = np.argmin(entropy, axis=1)

        # deduce the outputs
        neighborhoods_size = (bestk + self.k_min).astype("uint8")
        eigenvalues = self.eigs_to_test[range(len(bestk)),bestk,:]
        normals = self.normals_to_test[range(len(bestk)),bestk,:]

        return neighborhoods_size, eigenvalues, normals
