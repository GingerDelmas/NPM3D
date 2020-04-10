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

num_trials = 1  # number of times the process is conducted (on a different random sample)

num_points_train = 1000  # num points per label, per train file, per trial
num_points_test = 1000  # num points per label, per test file, per trial
k_min = 20  # minimum neighborhood size considered
k_max = 22  # maximum neighborhood size considered

# pick one among 'k_dummy', 'k_critical_curvature', 
# 'k_min_shannon_entropy' and 'k_min_eigenentopy'
k_chooser = 'k_min_eigenentopy'

################################################################################
# PATHS
################################################################################

data_dir = '../../NPM3D_local_files/data'
results_dir = '../../NPM3D_local_files/results'
save_dir = '../../NPM3D_local_files/saves'

################################################################################
# MAIN
################################################################################

if __name__ == '__main__':    

    # get the content of the data directories
    files = [f for f in os.listdir(data_dir) if f.endswith(('.ply', '.txt'))]
    
    # select the desired files, keeping the selection in the form of a list
    train_files = [files[0], files[1]]
    test_files = [files[0]]
    print("Files chosen:", "\nTrain:", train_files, "\nTest:", test_files)
    
    
    for trial in range(num_trials):
        print('\nTRIAL', trial)
        t0_trial = time.time()
        
        
        ######################
        # TRAIN
        ######################
        
        # variables keeping tally across multiple training sets
        norms = []
        selected_features = []
        train_feature_finders = []
        train_clouds = []
        train_indices = []
        
        print("\nTraining:")
        for file in train_files:
            
            print("\nProcessing", file)
            cloud_path = data_dir + '/' + file
            
            # case of bildstein files
            if file.endswith('.txt'):
                label_path = data_dir + '/' + file.sub('.txt', '.labels')
            else:
                label_path = None
    
            # create train cloud
            print('\nCollect and preprocess cloud')
            t0 = time.time()
            cloud = train_test_cloud(cloud_path, save_dir, label_path=label_path)
            t1 = time.time()
            cloud.sample_n_points_per_label(num_points_train, seed=trial)  # sample a training set
            indices = cloud.hand_sampled_points(cloud.train_samples_indices)  # get indices for training set
            cloud.get_split_statistics()  # print the statistics on the samples
            cloud.save()  # save progress
            t2 = time.time()
            print('Done in %.0f seconds' % (t2 - t0), 
                  '(without load: %.0f seconds)' % (t2 - t1 + cloud.compute_time))
            
            # calculate neighborhoods
            print("\nComputing neighborhoods")
            t0 = time.time()
            nf = neighborhood_finder(cloud, indices, save_dir, k_min=k_min, k_max=k_max)
            t1 = time.time()
            ks, eigs, normals = eval('nf.%s()' % k_chooser)  # find an optimal k value for each query
            nf.save()  # save progress
            t2 = time.time()
            print('Done in %.0f seconds' % (t2 - t0), 
                  '(without load: %.0f seconds)' % (t2 - t1 + nf.compute_time))
            
            # calculate features
            print("\nCompute features")
            t0 = time.time()
            ff = features_finder(cloud, indices, ks, eigs, normals, save_dir, 
                                 save_norm=True, use_norm=False, norm=None)
            t1 = time.time()
            ff.save()
            t2 = time.time()
            print('Done in %.0f seconds' % (t2 - t0), 
                  '(without load: %.0f seconds)' % (t2 - t1 + ff.compute_time))
            
            # select features
            print("\nFeature selection", end='')
            t0 = time.time()
            ff.feature_selection(results_dir=results_dir)
            t1 = time.time()
            print("\nSelected features : {}".format(ff.selected))
            print('Done in %.0f seconds' % (t1 - t0))
            
            # update tally variables
            norms.append(ff.ft_norm)
            selected_features.append(ff.selected)
            train_feature_finders.append(ff)
            train_clouds.append(cloud)
            train_indices.append(indices)
        
        # take the mean of norms selected features listed
        norms_all = {key: [sum(n[key][0] for n in norms), sum(n[key][1] for n in norms)] 
                     for key in norms[0].keys()}
        
        
        ######################
        # TEST
        ######################
        
        # variables keeping tally across multiple training sets
        test_feature_finders = []
        test_clouds = []
        test_indices = []
        
        print("\nTesting:")
        for file in test_files:
            
            print("\nProcessing", file)
            cloud_path = data_dir + '/' + file
            
            # create test cloud
            print('\nCollect and preprocess cloud')
            t0 = time.time()
            cloud = train_test_cloud(cloud_path, save_dir)
            t1 = time.time()
            cloud.sample_n_points_per_label(num_points_train, seed=trial, test=True)  # sample a test set
            cloud.get_split_statistics()  # print the statistics on the samples
            cloud.save()  # save progress
            t2 = time.time()
            print('Done in %.0f seconds' % (t2 - t0), 
                  '(without load: %.0f seconds)' % (t2 - t1 + cloud.compute_time))
            
            # calculate neighborhoods
            print("\nComputing neighborhoods")
            t0 = time.time()
            indices = cloud.hand_sampled_points(cloud.test_samples_indices)
            nf = neighborhood_finder(cloud, indices, save_dir, k_min=k_min, k_max=k_max)
            t1 = time.time()
            ks, eigs, normals = eval('nf.%s()' % k_chooser)  # find an optimal k value for each query
            nf.save()  # save progress
            t2 = time.time()
            print('Done in %.0f seconds' % (t2 - t0), 
                  '(without load: %.0f seconds)' % (t2 - t1 + nf.compute_time))
            
            # calculate features
            print("\nCompute features")
            t0 = time.time()
            ff = features_finder(cloud, indices, ks, eigs, normals, save_dir, 
                                 save_norm=False, use_norm=True, norm=norms_all)
            t1 = time.time()
            ff.save()
            t2 = time.time()
            print('Done in %.0f seconds' % (t2 - t0), 
                  '(without load: %.0f seconds)' % (t2 - t1 + ff.compute_time))
            
            # update tally variables
            test_feature_finders.append(ff)
            test_clouds.append(cloud)
            test_indices.append(indices)
            
            
        print("\nTrain Classifier")
        t0 = time.time()
        # take all selected features listed
        selected_features_all = list(set(sum(selected_features, [])))
        
        # update the selected features and extract them
        X_train = []
        for ff in train_feature_finders:
            ff.selected = selected_features_all
            X_train.append(ff.hand_features())
        
        X_test = []
        for ff in test_feature_finders:
            ff.selected = selected_features_all
            X_test.append(ff.hand_features())
        
        """
        clf = classifier(train_clouds, train_indices, test_indices, X_train, X_test, cloud_te=test_clouds)
        rf = clf.random_forest()
        y_pred, score = clf.evaluate(rf)
        t1 = time.time()
        print('Done in %.0f seconds' % (t1 - t0))
        print("Evaluation : {}% of points from the testing set were correctly classified.\n".format(np.round(score,2)*100))
        clf.get_classification_statistics(y_pred)
        """
    

        print("\nSaving Predictions as Cloud")
        """
        ### save result (train set, here)
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
        """

        t1_trial = time.time()
        print('Full trial done in %.0f seconds.\n' % (t1_trial - t0_trial))
    
