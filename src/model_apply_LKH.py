import os, json, time, sys

import torch
import numpy as np

from or_search import ormain
import utils
import model.net as net
import model.dataset as dataset
import beam_search
import tsplib95
import lkh

def array_to_str(array: np.ndarray):
    np.set_printoptions(threshold=sys.maxsize)
    str = np.array2string(array, separator=' ', max_line_width=sys.maxsize)
    str = str.replace('.','').replace(' [','').replace('[','').replace(']','')
    return str


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

model = net.RouteNetV5(router_embbed_dim=params.router_embbed_dim, num_routers=params.num_routers, num_heads=params.num_heads, contraction_factor=params.contraction_factor, dropout=params.dropout_rate)
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

print('Calculating Sequence...')
start = time.time()
invalid = 0
# Model Apply Output


# for i in range(1):
#     batch = next(iter(dataloader))
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
    
#     # OR search
#     max_time = 24*3600*100
#     df_dist = (-output)**2.4*1.46 # best 1
#     total_stops = len(df_dist)
#     # print (np.mean(df_dist))
#     sequence = ormain(df_dist, start_node, max_time, total_stops)
    
    # LKH search
    df_dist = (-output)**2.4*1.46
    total_stops = len(df_dist)
    LKH_input = f"""
NAME: RC
TYPE: VRPB
DIMENSION: {total_stops}
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
{array_to_str(np.array(df_dist).astype(int))}
EOF
"""
    problem = tsplib95.parse(LKH_input)
    solver_path = 'LKH'
    ans = lkh.solve(solver_path, 
              problem=problem, 
              INITIAL_TOUR_ALGORITHM = "WALK",
              POPULATION_SIZE = 100, 
              PRECISION = 10,
              VEHICLES = 1,
              TIME_LIMIT = 60,
              RUNS = 100,
              DEPOT = start_node + 1,
              TRACE_LEVEL = 1)
    sequence = [x-1 for x in ans[0]]
    
    if sequence == None:
        # Beam Search
        sequence = beam_search.beam_search(
            start_node=start_node, weight_matrix=np.exp(output)*50, num_beam=int(1*output.shape[0])
        ).tolist()
        
        output_dict = {
           stop_id:sequence.index(i) for i, stop_id in enumerate(stop_ids)
        }
        invalid += 1
    else: 
        output_dict = {
            stop_id:sequence.index(i) for i, stop_id in enumerate(stop_ids) # LKH
#            stop_id:sequence[i] for i, stop_id in enumerate(stop_ids) # OR-tools
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
end = time.time()
print("Elapsed = %s" % (end - start))
print ("Invalid Results (Revert to BeamSearch):", invalid)


# Finish Apply
print("Done!")
