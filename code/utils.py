#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31/03/2020

Projet NPM 3D

Jeffery Durand and Ginger Delmas

################################

Here are defined utilitary & general functions :
    - class saveable
    - function save_cloud_and_scalar_fields
"""
################################################################################
# IMPORTS
################################################################################

import numpy as np
import os, time, pickle, hashlib

from sklearn.neighbors import KDTree

from ply import write_ply

################################################################################
# CLASS DEFINITIONS
################################################################################

class saveable:
    """
        Enables subclasses to save and load. Takes:
            - save_dir : directory where we will save a class for later reuse
            - identifiers : list of attributes that make a class unique;
                    i.e. if the identifiers given match the identifiers of a
                    previously calculated class instance, the latter can be loaded
            - save_file : optional, name of the file where we will save the cloud

            Unless the save_file filename is given, the identifiers are hashed
            to generate a filename used for saving and loading
    """

    def __init__(self, save_dir, identifiers, save_file=None):
        # save the directory, file name, and path
        self.save_dir = save_dir
        if save_file is None:
            m = hashlib.md5()
            m.update(bytes(str(identifiers), 'utf8'))
            self.save_file = m.hexdigest()[:10]
        else:
            self.save_file = save_file
        self.save_path = self.save_dir + '/' + self.save_file

    def save(self, save_path=None):
        # save the file at the default or custom path
        if save_path is None:
            save_path = self.save_path
        with open(save_path, 'wb') as f:
            # print(" (progress saving) ", end='')
            pickle.dump(self.__dict__, f)


    def load(self, load_if_possible=True, save_path=None):
        # load a file from the default or custom path
        if save_path is None:
            save_path = self.save_path
        # return True if there is a file to load and we want to load it
        if load_if_possible and os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                # print(" (loading) ", end='')
                self.__dict__ = pickle.load(f)
            return True
        else:
            return False


################################################################################
# FUNCTIONS DEFINITIONS
################################################################################

def format_val(val):
    """
    Format arrays of percentages to print out.
    'val' is an array of floats or a number.
    """
    if isinstance(val, np.ndarray):
        ret = [str(u)[:5] for u in (np.round(val,2)*100)]
    else :
        ret = str((np.round(val,2)*100))[:5]
    return ret

def find_intersection(u):
    """
    In :
        - u : list containing lists of values
    Out :
        - v : list of values appearing in every sublist of 'u'
    """
    v = []
    for x in u[0]:
        found = True
        for L in u[1:]:
            found = found & (x in L)
        if found :
            v.append(x)
    return v

def arrange_in_matrix(considered_class, values, num_class):

    """
    In :
        - considered_class : indices of the classes appearing in the testing set, for each trial
        - values : values to sum according to the class
                   'values' is a list containing for each trial a list of values (one value per considered class)
        - num_class : number of class potentially encountered in the testing set

    Out :
        - array of shape (num_class), containing the sum of the values according to their class

    """

    num_trials = len(considered_class)
    matrix = np.zeros((num_trials, num_class))

    for trial in range(num_trials):
        matrix[trial, np.array(considered_class[trial])] = np.array(values[trial])

    return np.sum(matrix, axis=0)

def save_cloud_and_scalar_fields(cloud, scalar_fields, fields_name, save_dir, filename):
    """
        Save a cloud into a .ply file to visualize it into CloudCompare for example.

        In :
            - cloud : shape (number of points, 3) contains the point coordinates
            - scalar_field : list of scalar fields (each scalar field is an array of length 'number of points')
            - fields_name : list containing the names under which saving the scalar fields (same size as scalar_fields)
            - save_dir, filename : filename must end with ".ply"
    """

    # Avoid handling bad headers (scalar fields must of float type)
    for i in range(len(scalar_fields)):
        scalar_fields[i] = scalar_fields[i].astype(float)

    # Store results in a .ply file to visualize in CloudCompare
    write_ply(save_dir + '/' + filename, [cloud] + scalar_fields,
                                        ['x', 'y', 'z'] + fields_name)

    print("File saved !\n")
