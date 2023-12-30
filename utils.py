import os
import numpy as np
import torch
import torch.nn as nn
from STSGCN import stsgcn
import csv

def construct_model(config):
    module_type = config['module_type']
    act_type = config['act_type']
    temporal_emb = config['temporal_emb']
    spatial_emb = config['spatial_emb']
    use_mask = config['use_mask']
    batch_size = config['batch_size']

    num_of_vertices = config['num_of_vertices']
    num_of_features = config['num_of_features']
    points_per_hour = config['points_per_hour']
    num_for_predict = config['num_for_predict']
    adj_filename = config['adj_filename']
    id_filename = config['id_filename']
    if id_filename is not None:
        if not os.path.exists(id_filename):
            id_filename = None


def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''

    :param distance_df_filename:  str, path of the csv file contains edges information
    :param num_of_vertices: int, the number of vertices
    :param type_: str, {connectivity, distance}
    :param id_filename: str
    :return: np.ndarray, adjacency matrix
    '''
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}