import os, json
import multiprocessing

import torch
import numpy as np
# from ortools.constraint_solver import routing_enums_pb2
# from ortools.constraint_solver import pywrapcp

import utils
import model.net as net
import model.dataset as dataset
import beam_search

# Set number of working cpu threads    
# https://jdhao.github.io/2020/07/06/pytorch_set_num_threads/
# https://github.com/pytorch/pytorch/issues/7087
torch.set_num_threads(multiprocessing.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())

# Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_PATH = os.path.join(BASE_DIR, 'experiments/base_model/params.json') # TODO
MODEL_PATH = os.path.join(BASE_DIR, 'experiments/base_model/best.pth.tar') # TODO
# MODEL_PATH = os.path.join(BASE_DIR, 'data/model_build_outputs/model.json')  # TODO 
OUTPUT_PATH = os.path.join(BASE_DIR, "data/model_apply_outputs/proposed_sequences.json")


# Load Model
print('Load Model...')

params = utils.Params(PARAMS_PATH)
checkpoint = torch.load(MODEL_PATH)

model = net.RouteNetV4(router_embbed_dim=params.router_embbed_dim, num_routers=params.num_routers, dropout=params.dropout_rate)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

model = torch.jit.script(model)
# torch.jit.save(model, os.path.join(BASE_DIR, 'data/model_build_outputs/model_script.pt'))


# Load Input Data
print('Reading Input Data...')

datasets = dataset.get_dataset(["apply"], os.path.join(BASE_DIR, 'data'))
apply_dataset = datasets["apply"]
collate_fn = dataset.get_collate_fn(stage="apply", params=params)
dataloader = dataset.DataLoader(apply_dataset, batch_size=1, collate_fn=collate_fn)


# Model Apply Output
for batch in dataloader:

    # load batch data
    inputs = batch['inputs']        # (n, max_num_stops, max_num_stops, num_1d_features + num_2d_features)
    input_0ds = batch['input_0ds']  # (n, num_0d_features)
    masks = batch['masks']          # (n, max_num_stops, max_num_stops)
    route_id = batch["route_ids"][0]      # str
    station_id = batch['station_ids'][0]  # str
    stop_ids = batch['stop_ids'][0]       # str[]

    # prediction
    output = model(inputs, input_0ds, masks)
    output = output.squeeze(0).detach().numpy() # (num_stops, num_stops)

    # perform sequence search
    start_node = stop_ids.index(station_id)
    sequence = beam_search.beam_search(start_node=start_node, weight_matrix=np.exp(output)*50, num_beam=int(1*output.shape[0])).tolist()
    output_dict = {
       stop_id:sequence.index(i) for i, stop_id in enumerate(stop_ids)
    }

    # append to proposed_sequences.json
    with open(OUTPUT_PATH, "r+") as file:
        data = json.load(file)
        data.update({
            route_id:{
                'proposed': output_dict
            }
        })
        file.seek(0)
        json.dump(data, file)


# Finish Apply
print("Done!")
