#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31/03/2020

Projet NPM 3D

Jeffery Durand et Ginger Delmas
"""

import numpy as np
import os, time, pickle
from ply import write_ply, read_ply

from sklearn.neighbors import KDTree

class saveable:
    """ enables subclasses to save and load"""
    
    def __init__(self, save_dir, save_file):
        self.save_dir = save_dir
        self.save_file = save_file
        self.save_path = save_dir + '/' + save_file
    
    
    def save(self, save_path=None):
        if save_path is None: 
            save_path = self.save_path
        with open(save_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
            
            
    def load(self, save_path=None):
        if save_path is None: 
            save_path = self.save_path
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                self.__dict__ = pickle.load(f)
            return True
        else: 
            return False
            

class cloud(saveable):
    """ basic class for point clouds """
    
    def __init__(self, save_dir, save_file):
        super().__init__(save_dir, save_file)
        
        self.points = np.empty((0, 3))
        self.labels = np.empty(0, dtype=int)
        self.cloud_id = np.empty(0, dtype=int)
        self.trees = {}
        
        # the set categories of the data
        self.label_names = {0: 'Unclassified',
                            1: 'Ground',
                            2: 'Building',
                            3: 'Poles',
                            4: 'Pedestrians',
                            5: 'Cars',
                            6: 'Vegetation'}


class train_cloud(cloud):
    """ compile and preprocess train data """
    
    def __init__(self, train_dir, save_dir, save_file='train_cloud', load_if_possible=True,
                 num_points_per_label=500):
        
        super().__init__(save_dir, save_file)
        
        if load_if_possible and not self.load():
            self.get_train_points(train_dir)
            self.sample_n_points_per_label(num_points_per_label)
            self.save()
        
        
    def get_train_points(self, train_dir):
        # Get all the ply files in data folder
        ply_files = [f for f in os.listdir(train_dir) if f.endswith('.ply')]
        
        self.num_clouds = len(ply_files)
        
        # Loop over each training cloud
        for i, file in enumerate(ply_files):

            # Load Training cloud
            cloud_ply = read_ply(train_dir + '/' + file)
            file_points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
            file_labels = cloud_ply['class']
            
            # add to the recorded points
            self.points = np.append(self.points, file_points, axis=0)
            self.labels = np.append(self.labels, file_labels)
            self.cloud_id = np.append(self.cloud_id, np.full(len(file_points), i))
            
            # make a KDTree
            self.trees[i] = KDTree(self.points)
        
            
    def sample_n_points_per_label(self, num_points_per_label):
        # dictionary of sampled points
        self.samples = {}
        
        for label in self.label_names.keys():
            
            # skip 'Unclassified' points
            if self.label_names[label] == 'Unclassified': continue
            
            # randomly sample points corresponding to the given label
            label_indices = np.flatnonzero(self.labels == label)
            assert len(label_indices) > num_points_per_label, 'not enough points per label'
            sampled_indices = np.random.choice(label_indices, num_points_per_label, replace=False)
            
            # add to the recorded sampled points
            self.samples[label] = self.points[sampled_indices]
        
        return self.samples
            
    
class test_cloud(cloud):
    """ compile and preprocess test data """
    
    def __init__(self, test_path, save_dir, save_file='test_cloud', load_if_possible=True):
        
        super().__init__(save_dir, save_file)
        
        if load_if_possible and not self.load():
            self.get_test_points(test_dir)
            self.save()
        
    def get_test_points(self, test_path):
        # Load Training cloud
        cloud_ply = read_ply(test_path)
        self.points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        self.cloud_id = np.full(len(self.points), 0)
        self.trees[0] = KDTree(self.points)
        
        
class neighborhood:
    """ general class for a point's k nearest neighborhood """
    def __init(self, cloud, indices):
        self.indices = indices
        self.cloud = cloud
        self.cloud_id = cloud.cloud_id[indices[0]]
        self.k = len(indices)
    
    def points(self):
        return self.cloud.points[self.indices]
    

class neighborhood_finder(saveable):
    """ Given points and queries, find a neighborhood size """
    def __init__(self, points, queries, ):
        self.points = points
        self.queries = queries
        
        
    def k_with_min_entropy(entropy_func, k_min=10, k_max=100):
        for k in range(k_min, k_max):
            pass
        

class features_finder(saveable):
    """ Given points, queries, and neighborhoods, find features """
    def __init__(self, points, queries, neighborhoods):
        self.points = points
        self.queries = queries
        self.neighborhoods = neighborhoods
        
    def get_structure_tensors(self):
        for neighborhood in self.neighborhoods:
            pass
            
    def get_dimensionality_features():
        # sphericity, planarity, linearity
        pass


class classifier:
    """ container for the different classifiers we want to test """
    pass


if __name__ == '__main__':
    
    # file paths
    train_dir = '../../NPM3D_local_files/data/train'
    test_dir = '../../NPM3D_local_files/data/test'
    results_dir = '../../NPM3D_local_files/results'
    save_dir = '../../NPM3D_local_files/saves'
    
    print('Collect Training Set')
    t0 = time.time()
    tc = train_cloud(train_dir, save_dir)
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))