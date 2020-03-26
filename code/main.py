#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31/03/2020

Projet NPM 3D

Jeffery Durand et Ginger Delmas
"""

import numpy as np
import os, time, pickle
from ply import read_ply

from sklearn.neighbors import KDTree

class saveable:
    """ 
        Enables subclasses to save and load. Takes:
            - save_dir : directory where we will save the cloud for later reuse
            - save_file : name of the file where we will save the cloud
    """
    
    def __init__(self, save_dir, save_file):
        # save the directory, file name, and path
        self.save_dir = save_dir
        self.save_file = save_file
        self.save_path = save_dir + '/' + save_file
    
    
    def save(self, save_path=None):
        # save the file at the default or custom path
        if save_path is None: 
            save_path = self.save_path
        with open(save_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
            
            
    def load(self, load_if_possible=True, save_path=None):
        # load a file from the default or custom path
        if save_path is None: 
            save_path = self.save_path
        # return True if there is a file to load and we want to load it
        if load_if_possible and os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                self.__dict__ = pickle.load(f)
            return True
        else: 
            return False
            

class cloud(saveable):
    """ 
        Basic class for a point cloud. Takes:
            - ply_path : the path to the .ply file containing the cloud
            - save_dir and save_file : see "saveable" class
            - include_labels : boolean indicating if we should (or can) take labels
    """
    
    def __init__(self, ply_path, save_dir, save_file):
        
        # call the "saveable" class __init__()
        super().__init__(save_dir, save_file)
        
        # the set categories of the data
        self.label_names = {0: 'Unclassified',
                            1: 'Ground',
                            2: 'Building',
                            3: 'Poles',
                            4: 'Pedestrians',
                            5: 'Cars',
                            6: 'Vegetation'}
        
        # save path to ply file
        self.ply_path = ply_path
        
        
    def fetch_points(self, include_labels: bool):
        # read the ply file and store content
        cloud_ply = read_ply(self.ply_path)
        self.points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        if include_labels: self.labels = cloud_ply['class']
        # make the KD Tree
        self.tree = KDTree(self.points)
        

class train_cloud(cloud):
    """ 
        Basic class for a training point cloud. Takes:
            - ply_path, save_dir, and save_file: see "cloud" class
            - load_if_possible : if True, loads the previously saved version if possible
            - num_points_per_label : number of points to randomly sample per labeled class
    """
    
    def __init__(self, ply_path, save_dir, save_file, load_if_possible=True,
                 num_points_per_label=500):
        
        # call the "cloud" class __init__()
        super().__init__(ply_path, save_dir, save_file)
        
        # if load() succeeds, skip initializing
        if not self.load(load_if_possible):
            # include labels since this is the train set
            self.fetch_points(include_labels=True)
            self.sample_n_points_per_label(num_points_per_label)
        
            
    def sample_n_points_per_label(self, num_points_per_label):
        """
            Sample a fixed number of points per labeled class to:
                1) limit computation time
                2) avoid bias more common classes
        """
        
        # dictionary of sampled points
        self.samples_indices = {}
        
        for label in self.label_names.keys():
            
            # skip 'Unclassified' points
            if self.label_names[label] == 'Unclassified': continue
            
            # randomly sample points corresponding to the given label
            # choose as many as possible if there aren't enough points
            label_indices = np.flatnonzero(self.labels == label)
            try:
                sampled_indices = np.random.choice(label_indices, num_points_per_label, replace=False)
            except ValueError:
                sampled_indices = label_indices
                print("Warning: class '{}' only has {}/{} points".format(
                    self.label_names[label], len(label_indices), num_points_per_label))
            
            # add to the recorded sampled points
            self.samples_indices[label] = sampled_indices
        
        return self.samples_indices
            
    
class test_cloud(cloud):
    """ Basic class for a test point cloud """
    
    def __init__(self, ply_path, save_dir, save_file, load_if_possible=True):
        
        # call the "cloud" class __init__()
        super().__init__(ply_path, save_dir, save_file)
        
        # if load_if_possible is True and load() succeeds, skip initializing
        if not self.load(load_if_possible):
            # skip labels since this is the test set
            self.fetch_points(include_labels=False)
    
        
class neighborhood:
    """ general class for a point's neighborhood """
    
    def __init(self, cloud, indices, normal, eigenvalues):
        self.indices = indices
        self.cloud = cloud
        self.k = len(indices)
        self.normal = normal
        self.eigenvalues = eigenvalues
    
    def points(self):
        return self.cloud.points[self.indices]
    

class neighborhood_finder(saveable):
    """ 
        Given a cloud and query indices, find a neighborhood.  
        
        In: 
            - cloud : the relevant 'cloud' class
            - query_indices : indices of the points in the cloud of which we want neighborhoods
            - save_dir and save_file : see "saveable" class
        
        The different optimal neighbourhood finders which are
        k_critical_curvature(), k_min_shannon_entropy(), and k_min_eigenentopy()
        should return:
        
        Out:
            - neighborhoods : a list of length len(query_indices) of 'neighborhood' classes
    """
    
    def __init__(self, cloud, query_indices, save_dir, save_file):
        
        # call the "saveable" class __init__()
        super().__init__(save_dir, save_file)
        
        # store arguments
        self.cloud = cloud
        self.query_indices = query_indices
        
        # parameters set internally
        self.k_min = 10
        self.k_max = 100
        
        # calculate all needed data
        self.eigenvalues, self.normals = self.compute_over_k_range()
        
        # TO IMPLEMENT
        # save() with the given file name some hash of the cloud and query_indices
        
        
    def compute_over_k_range(self):
        """ 
            Loop through all considered k values and store needed data:
                - eigenvectors of the structure tensor l1, l2, l3
                - the normal vector found with this neghiborhood
                
            Vectorized by querying the knn for all query_indices at a given k
                
            Out: both outputs have shape (len(query_indices), k_max - k_min, 3)
                - eigenvalues : np array storing l1, l2, l3 for every value of query_indices and k,
                - normals : np array storing the normal vector found for every value of query_indices and k,                
        """
        
        # empty containers for the output
        eigenvalues = np.empty((len(self.query_indices), self.k_max - self.k_min, 3))
        normals = np.empty((len(self.query_indices), self.k_max - self.k_min, 3))
        
        for k in range(self.k_min, self.k_max):
            knn = self.cloud.tree.query(self.query_indices, k, return_distance=False)
            
            # TO IMPLEMENT
            # implement a call to features_finder to get the PCA, normals, etc
            
        return eigenvalues, normals
        
        
    def k_critical_curvature():
        """
            k maximizing the change in curvature C = l3 / (l1 + l2 + l3)
            where li is the ith biggest eigenvalue of the structure tensor
        """
        
        # TO IMPLEMENT
        neighborhoods = []
        
        return neighborhoods
    

    def k_min_shannon_entropy():
        """
            k minimizing the entropy Edim = - L*ln(L) - P*ln(P) - S*ln(S)
            where L, P and S are the linearity, planarity and sphericity
        """
        
        # TO IMPLEMENT
        neighborhoods =[]
        
        return neighborhoods
    
        
    def k_min_eigenentopy():
        """
            k minimizing the entropy El = - e1*ln(e1) - e2*ln(e2) - e2*ln(e2)
            where ei is the normalized ith biggest eigenvalue of the structure tensor
        """
        
        # TO IMPLEMENT
        neighborhoods = []
        
        return neighborhoods
        

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
    
    print('Collect and preprocess training sets')
    t0 = time.time()
    
    ply_files = [f for f in os.listdir(train_dir) if f.endswith('.ply')]
    
    for i, file in enumerate(ply_files):
        tc = train_cloud(train_dir + '/' + file, save_dir, 'train_cloud_{}'.format(i))
        tc.save()
        print("Processed {}/{} clouds: {}".format(i + 1, len(ply_files), file))
        
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))