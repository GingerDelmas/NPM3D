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
separate_cloud = False

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

        if separate_cloud: # the training and testing set come from different clouds

            ### parameters
            i_tr = 1
            i_te = 0
            file_tr = ply_files[i_tr]
            file_te = ply_files[i_te]
            nppl_train = 1000
            nppl_test = 3000
            unic_k = 20
            saveCloud = True
            load = False

            ### load cloud
            print('\nCollect and preprocess cloud')
            tc_tr = train_test_cloud(train_dir + '/' + file_tr, save_dir,
                            'train_cloud_{}_tr{}'.format(i_tr, nppl_train, 0),
                            load_if_possible=load,
                            num_points_per_label_train=nppl_train,
                            num_points_per_label_test=0)

            tc_tr.get_split_statistics()

            tc_te = train_test_cloud(train_dir + '/' + file_te, save_dir,
                            'test_cloud_{}_te{}'.format(i_te, 0, nppl_test),
                            load_if_possible=load,
                            num_points_per_label_train=0,
                            num_points_per_label_test=nppl_test)

            tc_te.get_split_statistics()

            if not load:
                tc_tr.save()
                tc_te.save()

            # hand sampled points
            train_indices = tc_tr.hand_sampled_points(tc_tr.train_samples_indices)
            test_indices = tc_te.hand_sampled_points(tc_te.test_samples_indices)

            ### find the right neighborhood (here : fixed)
            print("Compute neighborhoods\n")
            nf_train = neighborhood_finder(tc_tr, train_indices, save_dir, 'neighbors_train_{}_k_{}'.format(i_tr,unic_k),
                                    load_if_possible=load, k_min=unic_k, k_max=unic_k)
            nf_test = neighborhood_finder(tc_te, test_indices, save_dir, 'neighbors_test_{}_k_{}'.format(i_te,unic_k),
                                    load_if_possible=load, k_min=unic_k, k_max=unic_k)
            if not load :
                nf_train.save()
                nf_test.save()

            neighborhoods_size_tr, eigenvalues_tr, normals_tr = nf_train.k_dummy()
            neighborhoods_size_te, eigenvalues_te, normals_te = nf_test.k_dummy()

            ### compute features
            ff_tr = features_finder(tc_tr, train_indices,
                                neighborhoods_size_tr, eigenvalues_tr, normals_tr,
                                save_dir, 'features_train_{}_k_{}'.format(i_tr, unic_k),
                                save_norm=True, use_norm=False, norm=None)
            ff_te = features_finder(tc_te, test_indices,
                                neighborhoods_size_te, eigenvalues_te, normals_te,
                                save_dir, 'features_test_{}_k_{}'.format(i_te, unic_k),
                                save_norm=False, use_norm=True, norm=ff_tr.ft_norm)

            if not load :
                print("Compute training features\n")
                ff_tr.features_2D_bins()
                ff_tr.features_2D()
                ff_tr.features_3D()

                ### feature selection
                print("Do feature selection")
                ff_tr.feature_selection()
                print("... selected features : {} \n".format(ff_tr.selected))
                ff_tr.save()

                print("Compute testing features\n")
                ff_te.norm = ff_tr.ft_norm # use normalization from the training set
                ff_te.features_2D_bins()
                ff_te.features_2D()
                ff_te.features_3D()
                ff_te.save()

            else :
                ff_te.load()
                ff_tr.load()

            print("... selected features : {} \n".format(ff_tr.selected))

            ### classify
            print("Classify")
            X_train = ff_tr.hand_features()
            X_test = ff_te.hand_features(selected_specific=ff_tr.selected, compute_specific=True)
            clf = classifier(tc_tr, train_indices, test_indices, X_train, X_test, test_cloud_diff=True, cloud_te=tc_te)
            rf = clf.random_forest()
            y_pred, score = clf.evaluate(rf)
            print("... evaluation : {}% of points from the testing set were correctly classified.\n".format(np.round(score,2)*100))
            clf.get_classification_statistics(y_pred)

            ### save result (train set, here)
            if saveCloud:
                ft_list_tr, ft_names_tr = ff_tr.prepare_features_for_ply()
                ft_list_tr += [tc_tr.labels[train_indices]]
                ft_names_tr += [name_of_class_label]
                filename_tr = "cloud_wrap_train_{}_{}.ply".format(nppl_train, unic_k)
                save_cloud_and_scalar_fields(tc_tr.points[train_indices], ft_list_tr,
                                            ft_names_tr, results_dir, filename_tr)

                ft_list_te, ft_names_te = ff_te.prepare_features_for_ply()
                ft_list_te += [tc_te.labels[test_indices], y_pred]
                ft_names_te += [name_of_class_label, "predicted_class"]
                filename_te = "cloud_wrap_test_{}_{}.ply".format(nppl_test, unic_k)
                save_cloud_and_scalar_fields(tc_te.points[test_indices], ft_list_te,
                                            ft_names_te, results_dir, filename_te)

        else :

            ### parameters
            i = "bildstein"
            fileName = "bildstein_station5_xyz_intensity_rgb"
            cloud_path = "../../NPM3D_local_files/data/semantic-8/{}.txt".format(fileName)
            label_path = "../../NPM3D_local_files/data/semantic-8/sem8_labels_training/{}.labels".format(fileName)
            file_type = "txt"

            # i = 1
            # cloud_path = train_dir + '/' + ply_files[i]
            # file_type = "ply"

            nppl_train = 1000
            nppl_test = 3000
            unic_k = 20
            saveCloud = True
            load = False

            ### load cloud
            print('\nCollect and preprocess cloud')
            tc = train_test_cloud(cloud_path,
                            save_dir,
                            'train_cloud_{}_tr{}_te{}'.format(i, nppl_train, nppl_test),
                            file_type=file_type,
                            label_path=label_path,
                            load_if_possible=load,
                            num_points_per_label_train=nppl_train,
                            num_points_per_label_test=nppl_test)

            tc.get_split_statistics()

            if not load:
                tc.save()

            # hand sampled points
            train_indices = tc.hand_sampled_points(tc.train_samples_indices)
            test_indices = tc.hand_sampled_points(tc.test_samples_indices)

            ### find the right neighborhood (here : fixed)
            print("Compute neighborhoods\n")
            nf_train = neighborhood_finder(tc, train_indices, save_dir, 'neighbors_train_{}_k_{}'.format(i,unic_k),
                                    load_if_possible=load, k_min=unic_k, k_max=unic_k)
            nf_test = neighborhood_finder(tc, test_indices, save_dir, 'neighbors_test_{}_k_{}'.format(i,unic_k),
                                    load_if_possible=load, k_min=unic_k, k_max=unic_k)
            if not load :
                nf_train.save()
                nf_test.save()

            neighborhoods_size_tr, eigenvalues_tr, normals_tr = nf_train.k_dummy()
            neighborhoods_size_te, eigenvalues_te, normals_te = nf_test.k_dummy()

            ### compute features
            ff_tr = features_finder(tc, train_indices,
                                neighborhoods_size_tr, eigenvalues_tr, normals_tr,
                                save_dir, 'features_train_{}_k_{}'.format(i, unic_k),
                                save_norm=True, use_norm=False, norm=None)
            ff_te = features_finder(tc, test_indices,
                                neighborhoods_size_te, eigenvalues_te, normals_te,
                                save_dir, 'features_test_{}_k_{}'.format(i, unic_k),
                                save_norm=False, use_norm=True, norm=ff_tr.ft_norm)

            if not load :
                print("Compute training features\n")
                ff_tr.features_2D_bins()
                ff_tr.features_2D()
                ff_tr.features_3D()

                ### feature selection
                print("Do feature selection")
                ff_tr.feature_selection()
                print("... selected features : {} \n".format(ff_tr.selected))
                ff_tr.save()

                print("Compute testing features\n")
                ff_te.norm = ff_tr.ft_norm # use normalization from the training set
                ff_te.features_2D_bins()
                ff_te.features_2D()
                ff_te.features_3D()
                ff_te.save()

            else :
                ff_te.load()
                ff_tr.load()

            ### classify
            print("Classify")
            X_train = ff_tr.hand_features()
            X_test = ff_te.hand_features(selected_specific=ff_tr.selected, compute_specific=True)
            clf = classifier(tc, train_indices, test_indices, X_train, X_test)
            rf = clf.random_forest()
            y_pred, score = clf.evaluate(rf)
            print("... evaluation : {}% of points from the testing set were correctly classified.\n".format(np.round(score,2)*100))
            clf.get_classification_statistics(y_pred)

            if saveCloud:
                ft_list, ft_names = ff_tr.prepare_features_for_ply()
                ft_list += [tc.labels[train_indices], tc.labels[test_indices], y_pred]
                ft_names += [name_of_class_label, "ground_truth_test", "predicted_class"]
                filename = "cloud_wrap_{}_{}.ply".format(nppl_train, unic_k)
                save_cloud_and_scalar_fields(tc.points[train_indices], ft_list,
                                            ft_names, results_dir, filename)

    else :
        for i, file in enumerate(ply_files):
            tc = train_test_cloud(train_dir + '/' + file, save_dir, 'train_cloud_{}'.format(i))
            tc.save()
            print("Processed {}/{} clouds: {}".format(i + 1, len(ply_files), file))


    t1 = time.time()
    print('Done in %.3f seconds.\n' % (t1 - t0))
