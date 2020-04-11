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

num_trials = 10  # number of times the process is conducted (on a different random sample)

num_points_train = 1000  # num points per label, per train file, per trial
num_points_test = 3000  # num points per label, per test file, per trial
k_min = 10  # minimum neighborhood size considered
k_max = 100  # maximum neighborhood size considered

# pick one among 'k_dummy', 'k_critical_curvature',
# 'k_min_shannon_entropy' and 'k_min_eigenentropy'
k_chooser = 'k_dummy'

all_features = False # if False, use only relevant features

################################################################################
# PATHS
################################################################################

data_dir = '../../NPM3D_local_files/data/Lille-Dijon'
results_dir = '../../NPM3D_local_files/results'
save_dir = '../../NPM3D_local_files/saves'

################################################################################
# MAIN
################################################################################

if __name__ == '__main__':

    # get the content of the data directories
    files = [f for f in os.listdir(data_dir) if f.endswith(('.ply', '.txt'))]

    # select the desired files, keeping the selection in the form of a list
    train_files = [files[2]]
    test_files = [files[2]]
    print("Files chosen:", "\nTrain:", train_files, "\nTest:", test_files)

    # NOTE : the different files used here are required to share the same
    #        labeling system

    # keep statistics over the different trials
    best_accuracy = 0 # to keep only the output results of the best trial
    all_trials_accuracy = []
    all_trials_selected_features = []
    all_trials_considered_labels = []
    all_trials_recall_by_class = []
    all_trials_precision_by_class = []
    all_trials_F_by_class = []
    all_trials_mean_recall = []
    all_trials_mean_precision = []
    all_trials_global_F = []
    all_trials_computation_time = []

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

        print("\nProcess training clouds:")
        for file in train_files:

            print("\nProcessing", file)
            cloud_path = data_dir + '/' + file

            # case of semantic-8 files
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
            cloud.get_split_statistics()  # print the statistics on the samples
            cloud.save()  # save progress
            t2 = time.time()
            print('Done in %.0f seconds' % (t2 - t0),
                  '(without load: %.0f seconds)' % (t2 - t1 + cloud.compute_time))

            # calculate neighborhoods
            print("\nComputing neighborhoods")
            t0 = time.time()
            indices = cloud.hand_sampled_points(cloud.train_samples_indices)  # get indices for training set
            nf = neighborhood_finder(cloud, indices, save_dir, k_min=k_min, k_max=k_max)
            t1 = time.time()
            ks, eigs, normals = eval('nf.%s()' % k_chooser)  # find an optimal k value for each query
            nf.save()  # save progress
            t2 = time.time()
            print('Done in %.0f seconds' % (t2 - t0),
                  '(without load: %.0f seconds)' % (t2 - t1 + nf.compute_time))

            if k_chooser != "k_dummy":
                plt.hist(ks, bins=np.max(ks)-np.min(ks))
                plt.savefig(results_dir + "/hist.png")
                plt.close()

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
            if all_features:
                ff.selected = list(ff.features.keys())
            else :
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
        norms_all = {key: [np.mean([float(n[key][0]) for n in norms]),
                           np.mean([float(n[key][1]) for n in norms])]
                     for key in norms[0].keys()}


        ######################
        # TEST
        ######################

        # variables keeping tally across multiple training sets
        test_feature_finders = []
        test_clouds = []
        test_indices = []

        print("\nProcess testing clouds:")
        for file in test_files:

            print("\nProcessing", file)
            cloud_path = data_dir + '/' + file

            # create test cloud
            print('\nCollect and preprocess cloud')
            t0 = time.time()
            cloud = train_test_cloud(cloud_path, save_dir)
            t1 = time.time()
            cloud.sample_n_points_per_label(num_points_test, seed=trial, test=True)  # sample a test set
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


        ######################
        # CLASSIFY
        ######################

        print("\nTrain Classifier")
        t0 = time.time()

        # take all selected features listed
        selected_features_all = list(set(sum(selected_features, [])))

        # update the selected features and extract them
        X_train = []
        for ff in train_feature_finders:
            ff.selected = selected_features_all
            X_train.append(ff.hand_features())
        X_train = np.concatenate(X_train, axis=0)

        X_test = []
        for ff in test_feature_finders:
            ff.selected = selected_features_all
            X_test.append(ff.hand_features())
        X_test = np.concatenate(X_test, axis=0)

        # extract labels of every considered points
        y_train = []
        for cloud, indices in zip(train_clouds, train_indices):
            y_train.append(cloud.labels[indices])
        y_train = np.concatenate(y_train)

        y_test = []
        for cloud, indices in zip(test_clouds, test_indices):
            y_test.append(cloud.labels[indices])
        y_test = np.concatenate(y_test)

        # eventually train the classifier, predict and evaluate labels
        # (here, 'cloud' is the last element of test_clouds)
        clf = classifier(X_train, X_test, y_train, y_test, cloud.label_names)
        rf = clf.random_forest()
        y_pred, measures = clf.evaluate(rf, results_dir)
        t1 = time.time()
        print('Done in %.0f seconds' % (t1 - t0))

        print("Evaluation : {}% of points from the testing set were correctly classified.\n".format(np.round(measures["accuracy"],2)*100))
        mess = "Other available measures (considered classes : {}): \n\t- recall by class (%) : {}\n\t- precision by class (%) : {}\n\t- F by class (%) : {}\n\t- mean recall : {}%\n\t- mean precision : {}%\n\t- global F : {}%"

        print(mess.format("'"+"', '".join([cloud.label_names[l] for l in measures["considered_labels"]])+"'",
                            format_val(measures["recall_by_class"]),
                            format_val(measures["precision_by_class"]),
                            format_val(measures["F_by_class"]),
                            format_val(measures["mean_recall"]),
                            format_val(measures["mean_precision"]),
                            format_val(measures["global_F"])))

        clf.get_classification_statistics(y_pred)

        print("\nSaving Predictions as Cloud")

        begin_y_pred = 0
        for f, ff, cloud, indices in zip(test_files, test_feature_finders, test_clouds, test_indices):
            # convert features to scalar field
            ft_list, ft_names = ff.prepare_features_for_ply()
            # build the scalar field for predicted classes
            ft_list += [cloud.labels[indices], y_pred[begin_y_pred:begin_y_pred+len(indices)]]
            begin_y_pred += len(indices) # for the next cloud
            ft_names += ["label", "predicted_class"]
            # also consider the neighborhood size as a scalar field
            ft_list += [ff.neighborhoods_size]
            ft_names += ["neighborhood_size"]
            # save cloud
            filename = "cloud_{}_predictions.ply".format(f[:-4])
            save_cloud_and_scalar_fields(cloud.points[indices], ft_list,
                                            ft_names, results_dir, filename)

        t1_trial = time.time()
        print('Full trial done in %.0f seconds.\n' % (t1_trial - t0_trial))

        #########################
        # SAVE TRIAL STATISTICS
        #########################

        # if this is the best model obtained so far, save generated results
        # (images & prediction cloud) under a different name, else erase them
        res_files = [f for f in os.listdir(results_dir) if f.endswith(('.ply', '.png'))]

        if measures["accuracy"] > best_accuracy :
            best_accuracy = measures["accuracy"]

            for r in res_files :
                if "best" not in r: # this will erase the previous "best" files (= update)
                    os.rename(results_dir+"/"+r, results_dir+"/"+"best_"+r)

        else : # erase every file that isn't representing the best result
            for r in res_files :
                if "best" not in r:
                    os.remove(results_dir+"/"+r)

        # store measured values to later compute averages and summarizations
        all_trials_accuracy.append(measures["accuracy"])
        all_trials_selected_features.append(selected_features_all)
        all_trials_considered_labels.append(measures["considered_labels"])
        all_trials_recall_by_class.append(measures["recall_by_class"])
        all_trials_precision_by_class.append(measures["precision_by_class"])
        all_trials_F_by_class.append(measures["F_by_class"])
        all_trials_mean_recall.append(measures["mean_recall"])
        all_trials_mean_precision.append(measures["mean_precision"])
        all_trials_global_F.append(measures["global_F"])
        all_trials_computation_time.append(t1_trial - t0_trial)

    #################################
    # COMPUTE ALL TRIALS STATISTICS
    #################################

    print("#########################")
    print("# All trials statistics #")
    print("#########################")

    print("\nFeatures selected at each trial :", find_intersection(all_trials_selected_features))
    print("Features selected over the different trials :", list(set(sum(all_trials_selected_features, []))))
    print("Mean number of feature per trial :", np.mean([len(tsf) for tsf in all_trials_selected_features]))
    print("Mean accuracy :", format_val(np.mean(all_trials_accuracy)), "%")
    print("Mean recall :", format_val(np.mean(all_trials_mean_recall)), "%")
    print("Mean precision :", format_val(np.mean(all_trials_mean_precision)), "%")
    print("Mean global F-measure :", format_val(np.mean(all_trials_global_F)), "%")
    print("Mean time computation :  %.0f seconds.\n" % np.mean(all_trials_computation_time))

    num_class = len(cloud.label_names)
    considered_labels = sum(all_trials_considered_labels, [])
    count_considered = np.array([considered_labels.count(l) for l in range(num_class)])
    count_considered[count_considered==0] = -1

    val1 = arrange_in_matrix(all_trials_considered_labels,all_trials_recall_by_class, num_class)/count_considered
    val2 = arrange_in_matrix(all_trials_considered_labels,all_trials_precision_by_class, num_class)/count_considered
    val3 = arrange_in_matrix(all_trials_considered_labels,all_trials_F_by_class, num_class)/count_considered

    d_name = max([len(cloud.label_names[l]) for l in cloud.label_names.keys()]+[len("class")])+3
    header = "{0:<%d}    recall   precision   F-measure\n" % (d_name)
    lineprint =   "{0:<%d} : {1:>5}%%   {2:>5}%%      {3:>5}%%" % (d_name)
    print(header.format("class"))
    for c in cloud.label_names.keys():
        print(lineprint.format(cloud.label_names[c], format_val(val1[c]), format_val(val2[c]), format_val(val3[c])))

    print("")
