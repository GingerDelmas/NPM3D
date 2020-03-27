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
            list with n elements, each being a numpy arrays of size len(query_indices)

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

        return [linearity, planarity, sphericity]


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

        # define useful function here
        vec2mat = lambda v : np.outer(v,v)

        eigenvalues = np.zeros((len(self.query_indices), 2))
        radius = np.zeros(len(self.query_indices))

        for ind, q in enumerate(self.query_indices):

            # use the closest neighbors based on 3D distance
            knn = self.cloud.tree.query(self.cloud.points[q].reshape(1,-1),
                                              self.neighborhoods_size[ind],
                                              return_distance=False)[0]

            # keep 2D coordinates only
            pts = self.cloud.points[knn,:2]

            ## radius
            # compute the 2D distance
            dist = np.linalg.norm(pts-self.cloud.points[q,:2], axis=1)
            radius[ind] = np.max(dist)

            ## compute the eigenvalues, while we're at it
            centroid = np.mean(pts, axis=0)

            # compute the covariance matrix
            cov = np.apply_along_axis(vec2mat, 1, pts - centroid)
            cov = sum(cov)/len(cov)

            # compute the eigenvalues and eigenvectors
            eigenvalues[ind], _ = np.linalg.eigh(cov)

        # local point density
        local_point_density = (self.neighborhoods_size+1) / (np.pi * radius**2)

        # sum & ratio
        lbda1, lbda2 = eigenvalues[:,1], eigenvalues[:,0]

        summ = lbda1 + lbda2
        ratio = lbda2 / lbda1

        return [radius, local_point_density, summ, ratio]

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

        eps = 10**(-5) # to avoid errors when eigenvalues = 0 (denominator, log)

        #### get the eigenvalues and normalize them
        e1, e2, e3 = self.eigenvalues[:,2], self.eigenvalues[:,1], self.eigenvalues[:,0]

        e1 /= e1 + e2 + e3
        e2 /= e1 + e2 + e3
        e3 /= e1 + e2 + e3

        #### geometric 3D properties

        # absolute height (no real need to store it...)
        absolute_height = self.cloud.points[self.query_indices,2]

        # radius, maximum height difference, height standard deviation
        radius = np.zeros(len(self.query_indices))
        max_height_diff = np.zeros(len(self.query_indices))
        height_std = np.zeros(len(self.query_indices))

        for ind,q in enumerate(self.query_indices):
            dist, knn = self.cloud.tree.query(self.cloud.points[q].reshape(1,-1),
                                              self.neighborhoods_size[ind],
                                              return_distance=True)
            # there is only one sample
            dist = dist[0]
            knn = knn[0]

            radius[ind] = np.max(dist)

            heights = self.cloud.points[knn,2]

            # QUESTION : is height difference defined wrt the query point ? Should it be negative (if possible) ?
            # max_height_diff[ind] = np.max(abs(heights - self.cloud[q])) # option 1
            max_height_diff[ind] = np.max(heights) - np.min(heights) # option 2

            height_std[ind] = np.std(heights)

        # local point density
        local_point_density = (self.neighborhoods_size+1) / (4./3. * np.pi * radius**3)

        # verticality
        verticality = 1 - self.normals[:,2]

        #### dimensionality features

        linearity = 1 - e2 / (e1 + eps)
        planarity = (e2 - e3) / (e1 + eps)
        sphericity = e3 / (e1 + eps) # = scattering

        ### other 3D shape features

        omnivariance = (e1*e2*e3)**(1/3.)
        anisotropy = (e1 - e3) / (e1 + eps)
        eigenentropy = - e1*np.log(e1 + eps) - e2*np.log(e2 + eps) - e3*np.log(e3 + eps)
        summ = e1 + e2 + e3
        curvature_change = e3 / (e1 + e2 + e3 + eps)

        return [absolute_height, radius, max_height_diff, height_std, local_point_density,verticality, linearity, planarity, sphericity, omnivariance,anisotropy, eigenentropy, summ, curvature_change]

    def features_2D_bins(self, side_length=0.20):
        """
            Return 2D accumulation map features:
                - number of points in bin
                - maximum height difference in bin
                - height standard deviation in bin
        """

        # Look up for the points of the 2D-projected cloud falling into the same bin as the sampled points.
        # ie : the KDtree is performed on the cloud,
        # and the query is performed on the center of the cells containing the sampled points.
        # The chebyshev metric is used to represent a cell, given its center, to find which are the points from
        # the cloud being in the so-called cell.

        # a) for each sampled (and 2D-projected) point, compute the center of the cell it falls in
        centers = ((self.cloud.points[self.query_indices,:2] / side_length).astype(int) + 1
                        ) * side_length + side_length / 2.

        # b) find the points from the 2D-projected cloud falling into the same cells
        treeBins = KDTree(self.cloud.points[:,:2], metric="chebyshev")
        ind = treeBins.query_radius(centers, r=side_length/2.)

        # c) compute features
        nb_points_in_bin = np.array([len(ind_q) for ind_q in ind])
        max_height_diff = np.array([np.max(self.cloud.points[ind_q,2])
                                    - np.min(self.cloud.points[ind_q,2])
                                    if nb_points_in_bin[i]>0 else 0
                                    for i,ind_q in enumerate(ind)])
        height_std = np.array([np.std(self.cloud.points[ind_q,2])
                                if nb_points_in_bin[i]>0 else 0
                                for i,ind_q in enumerate(ind)])

        return [nb_points_in_bin, max_height_diff, height_std]
