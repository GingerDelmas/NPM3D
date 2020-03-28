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
# MAIN
################################################################################

if __name__ == '__main__':

    # file paths
    train_dir = '../../NPM3D_local_files/mini_data'
    test_dir = '../../NPM3D_local_files/data/test'
    results_dir = '../../NPM3D_local_files/results'
    save_dir = '../../NPM3D_local_files/saves'

    t0 = time.time()

    ply_files = [f for f in os.listdir(train_dir) if f.endswith('.ply')]

    if developpement:

        ### parameters
        i = 0
        file = ply_files[i]
        nppl_train = 1000
        nppl_test = 3000
        unic_k = 20
        saveCloud = False
        load = True

        ### load cloud
        print('Collect and preprocess training sets')
        tc = train_test_cloud(train_dir + '/' + file, save_dir,
                        'train_cloud_{}_tr{}_te{}'.format(i, nppl_train, nppl_test),
                        load_if_possible=load,
                        num_points_per_label_train=nppl_train,
                        num_points_per_label_test=nppl_test)

        tc.get_statistics()

        if not load:
            tc.save()

        # hand sampled points
        query_indices = np.concatenate([tc.train_samples_indices[label] for label in tc.train_samples_indices.keys()])

        ### find the right neighborhood (here : fixed)
        print("Compute neighborhoods")
        nf = neighborhood_finder(tc, query_indices, save_dir, 'neighbors_{}'.format(i),
                                load_if_possible=load, k_min=unic_k, k_max=unic_k)
        if not load :
            nf.save()

        neighborhoods_size, eigenvalues, normals = nf.k_dummy()

        ### compute features
        print("Compute features")
        ff = features_finder(tc, query_indices,
                            neighborhoods_size, eigenvalues, normals,
                            save_dir, 'features_{}'.format(i))

        if not load :
            ff.features_dim()
            ff.features_2D_bins()
            ff.save()
        else :
            ff.load()

        ### feature selection
        print("Do feature selection")
        ff.feature_selection()
        print("-> selected features : {}".format(ff.selected))

        ### classify
        print("Classify")
        X = ff.hand_features()

        ### save result
        if saveCloud:
            ft_list, ft_names = ff.prepare_features_for_ply()
            filename = "cloud_wrap_{}_{}.ply".format(num_points_per_label, unic_k)
            save_cloud_and_scalar_fields(tc.points[query_indices], ft_list,
                                        ft_names, results_dir, filename)

    else :
        for i, file in enumerate(ply_files):
            tc = train_test_cloud(train_dir + '/' + file, save_dir, 'train_cloud_{}'.format(i))
            tc.save()
            print("Processed {}/{} clouds: {}".format(i + 1, len(ply_files), file))


    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))
