# OR-only 
from os import path
import sys, json, time
import datetime

import numpy as np
import pandas as pd

# from ortools.constraint_solver import routing_enums_pb2
# from ortools.constraint_solver import pywrapcp

# ML
import os
import json
import random
import collections
import enum
import typing

from tqdm import tqdm
from scipy import spatial
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import argparse
import logging
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
import model.net as net
import model.dataset as dataset
import model.loss as loss
from beam_search import *


## Initialize Paths
sys.argv = ['']
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(" ")))
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=os.path.join(BASE_DIR, 'data'),
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default=os.path.join(BASE_DIR, 'data/model_build_outputs'),
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')
model_path = os.path.join(args.model_dir, 'best.pth.tar')


new_routes_path = path.join(BASE_DIR, "data/model_apply_inputs/new_route_data.json")
df_route = pd.read_json(new_routes_path).transpose()



## Load Data
datasets = dataset.get_dataset(["apply"], args.data_dir)
build_dataset = datasets["apply"]

## Load Model
params = utils.Params(json_path)
model = net.RouteNetV4(router_embbed_dim=params.router_embbed_dim, num_routers=params.num_routers, dropout=params.dropout_rate)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

## Load Evaluation Dataset
train_set= build_dataset
train_sampler = dataset.BucketSampler([route.num_stops for route in train_set], batch_size=1, shuffle=False, drop_last=False)
collate_fn = dataset.get_collate_fn(stage="apply", params=params)
train_loader = dataset.DataLoader(train_set, batch_sampler=train_sampler, collate_fn=collate_fn)



## Evaluation
output_dict = {}
iterator = iter(train_loader)
for i in range(len(train_loader)):
    batch = next(iterator)

    inputs = batch['inputs'] # (n, max_num_stops, max_num_stops, num_1d_features + num_2d_features)
    input_0ds = batch['input_0ds'] # (n, num_0d_features)
    masks = batch['masks'] # (n, max_num_stops, max_num_stops)
    route_ids = batch["route_ids"][0]

    print(route_ids)

    outputs = model(inputs, input_0ds, masks)
    output = outputs.squeeze().detach().numpy()


    #### get route_no and sequence
    route_no = route_ids
    keys = df_route.loc[route_no]["stops"].keys()
    mydict = df_route.loc[route_no]["stops"]
    for _ in mydict.keys():
        if mydict[_]["type"] == "Station":
            station_code = _
    station_no = list(keys).index(station_code)
    stop_id_map = list(keys)
    ####


    # Compute output ranking
    starting_node = station_no
    sequence = beam_search(start_node=starting_node, weight_matrix=np.exp(output)*50, num_beam=int(1*output.shape[0])).tolist()
    rank = [0]*len(sequence)
    for i in range(len(rank)):
        rank[i] = sequence.index(i)

    # zip to dict 

    values_list = rank
    zip_itr = zip(stop_id_map, rank)
    a_dict = dict(zip_itr)
    a_dict = dict(proposed = a_dict)
    output_dict[route_no] = a_dict


# Write output data
output_path = path.join(BASE_DIR, "data/model_apply_outputs/proposed_sequences.json")
with open(output_path, "w") as out_file:
    json.dump(output_dict, out_file)
print("Done!")
