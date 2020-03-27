#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31/03/2020

Projet NPM 3D

Jeffery Durand and Ginger Delmas
"""
################################################################################
# IMPORTS
################################################################################

import numpy as np
import os, time, pickle
from ply import read_ply, write_ply

from sklearn.neighbors import KDTree

from utils import *
from cloud_env import *
from neighborhood import *
from feature import *

################################################################################
# GLOBAL VARIABLES
################################################################################

# specify wether some calculus must be lighten (eg : work on one cloud only)
developpement = True

################################################################################
# CLASS DEFINITIONS
################################################################################

class classifier:
    """ container for the different classifiers we want to test """
    pass

################################################################################
# MAIN
################################################################################

if __name__ == '__main__':

    # file paths
    train_dir = '../../NPM3D_local_files/mini_data'
    test_dir = '../../NPM3D_local_files/data/test'
    results_dir = '../../NPM3D_local_files/results'
    save_dir = '../../NPM3D_local_files/saves'

    print('Collect and preprocess training sets')
    t0 = time.time()

    ply_files = [f for f in os.listdir(train_dir) if f.endswith('.ply')]

    if developpement:

        # parameters
        i = 0
        file = ply_files[i]
        num_points_per_label = 3000
        unic_k = 20
        tested_features="2D-features" # "2D-bins", "feature-dim"

        # load cloud
        tc = train_cloud(train_dir + '/' + file, save_dir, 'train_cloud_{}_{}'.format(i, num_points_per_label),
                        load_if_possible=True, num_points_per_label=num_points_per_label)

        # hand sampled points
        query_indices = np.concatenate([tc.samples_indices[label] for label in tc.samples_indices.keys()])

        # find the right neighborhood (here : fixed)
        nf = neighborhood_finder(tc, query_indices, save_dir, 'neighbors_{}'.format(i),
                                k_min=unic_k, k_max=unic_k)
        neighborhoods_size, eigenvalues, normals = nf.k_dummy()

        # compute features
        ff = features_finder(tc, query_indices,
                            neighborhoods_size, eigenvalues, normals,
                            save_dir, 'features_{}'.format(i))

        if tested_features=="feature-dim":
            ft_list = ff.features_dim()
            # save result
            filename = "cloud_ftdim_{}_{}.ply".format(num_points_per_label, unic_k)
            save_cloud_and_scalar_fields(tc.points[query_indices], ft_list,
                                        ["linearity", "planarity", "sphericity"],
                                        save_dir, filename)

        elif tested_features=="2D-bins":
            ft_list = ff.features_2D_bins()

            # save result
            filename = "cloud_ft2Dbins_{}_{}.ply".format(num_points_per_label, unic_k)
            save_cloud_and_scalar_fields(tc.points[query_indices], ft_list,
                                        ["nbPtsInBin", "maxHeightDiff", "heightStd"],
                                        save_dir, filename)

        elif tested_features=="2D-features":
            ft_list = ff.features_2D()

            # save result
            filename = "cloud_ft2D_{}_{}.ply".format(num_points_per_label, unic_k)
            save_cloud_and_scalar_fields(tc.points[query_indices], ft_list,
                                        ["radius", "local_point_density", "sum", "ratio"],
                                        save_dir, filename)

    else :
        for i, file in enumerate(ply_files):
            tc = train_cloud(train_dir + '/' + file, save_dir, 'train_cloud_{}'.format(i))
            tc.save()
            print("Processed {}/{} clouds: {}".format(i + 1, len(ply_files), file))


    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))
