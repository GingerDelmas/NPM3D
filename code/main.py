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
from classifier import *

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
        train_indices = tc.hand_sampled_points(tc.train_samples_indices)
        test_indices = tc.hand_sampled_points(tc.test_samples_indices)

        ### find the right neighborhood (here : fixed)
        print("Compute neighborhoods")
        nf_train = neighborhood_finder(tc, train_indices, save_dir, 'neighbors_train_{}'.format(i),
                                load_if_possible=load, k_min=unic_k, k_max=unic_k)
        nf_test = neighborhood_finder(tc, test_indices, save_dir, 'neighbors_test_{}'.format(i),
                                load_if_possible=load, k_min=unic_k, k_max=unic_k)
        if not load :
            nf_train.save()
            nf_test.save()

        neighborhoods_size_tr, eigenvalues_tr, normals_tr = nf_train.k_dummy()
        neighborhoods_size_te, eigenvalues_te, normals_te = nf_test.k_dummy()

        ### compute features
        print("Compute features")
        ff_tr = features_finder(tc, train_indices,
                            neighborhoods_size_tr, eigenvalues_tr, normals_tr,
                            save_dir, 'features_train_{}'.format(i))
        ff_te = features_finder(tc, test_indices,
                            neighborhoods_size_te, eigenvalues_te, normals_te,
                            save_dir, 'features_test_{}'.format(i))

        if not load :
            ff_tr.features_dim()
            ff_tr.features_2D_bins()
            ff_tr.save()

            ff_te.features_dim()
            ff_te.features_2D_bins()
            ff_te.save()

        else :
            ff_tr.load()
            ff_te.load()

        ### feature selection
        print("Do feature selection")
        ff_tr.feature_selection()
        print("... selected features : {}".format(ff_tr.selected))

        ### classify
        print("Classify")
        X_train = ff_tr.hand_features()
        X_test = ff_te.hand_features(selected_specific=ff_tr.selected, compute_specific=True)
        clf = classifier(tc, train_indices, test_indices, X_train, X_test)
        rf = clf.random_forest()
        y_pred, score = clf.evaluate(rf)
        print("... evaluation : {}% of points from the testing set were correctly classified.".format(np.round(score,2)*100))

        ### save result (train set, here)
        if saveCloud:
            ft_list, ft_names = ff_tr.prepare_features_for_ply()
            filename = "cloud_wrap_{}_{}.ply".format(num_points_per_label, unic_k)
            save_cloud_and_scalar_fields(tc.points[train_indices], ft_list,
                                        ft_names, results_dir, filename)

    else :
        for i, file in enumerate(ply_files):
            tc = train_test_cloud(train_dir + '/' + file, save_dir, 'train_cloud_{}'.format(i))
            tc.save()
            print("Processed {}/{} clouds: {}".format(i + 1, len(ply_files), file))


    t1 = time.time()
    print('Done in %.3f seconds.\n' % (t1 - t0))
