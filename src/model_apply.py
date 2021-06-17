import os, json
import multiprocessing
import typing
import asyncio

import torch
import numpy as np
import aiofiles as aiof


import utils
import model.net as net
import model.dataset as dataset
import model.search as search

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

model = net.RouteNetV5(router_embbed_dim=params.router_embbed_dim, num_routers=params.num_routers, num_heads=params.num_heads, num_groups=params.num_groups, contraction_factor=params.contraction_factor, dropout=params.dropout_rate)
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
tasks : typing.List[search.Task]= []

# loop through dataloader
for batch in dataloader:

    # load batch data
    inputs = batch['inputs']        # (n, max_num_stops, max_num_stops, num_1d_features + num_2d_features)
    input_0ds = batch['input_0ds']  # (n, num_0d_features)
    masks = batch['masks']          # (n, max_num_stops, max_num_stops)
    route_id = batch["route_ids"][0]      # str
    station_id = batch['station_ids'][0]  # str
    stop_ids = batch['stop_ids'][0]       # str[]
    num_stops = batch['num_stops'][0]     # int
    
    # prediction
    output = model(inputs, input_0ds, masks)  # (num_stops, num_stops)
    output = output.squeeze(0).detach().numpy()

    # append to tasks list
    tasks.append(search.Task(
        route_id=route_id,
        stop_ids=stop_ids,
        station_id=station_id,
        num_stops=num_stops,
        input = inputs.squeeze(0).detach().numpy(),
        output = output.squeeze(0).detach().numpy()
    ))


# Function to solve task
async def solve(task: search.Task):

    # unpack arrays from task
    MAX_TIME = 24*3600*100
    route_id = task.route_id
    output = task.output
    total_stops = task.num_stops
    start_node = task.stop_ids.index(task.station_id)

    # perform sequence search
    # df_dist = (1/np.exp(output))
    # df_dist = (-output)**2.4 # best 2
    # df_dist = (-output)**2.4*1.46 # best 1
    df_dist = (-output)**2.4
    total_stops = len(df_dist)
    
    # or search TODO ERROR_BOUND
    sequence = search.or_search(df_dist, start_node, MAX_TIME, total_stops)
    
    # generate sequence
    if sequence is None:
        # Beam Search
        sequence = search.beam_search(
            start_node=start_node, weight_matrix=np.exp(output)*50, num_beam=int(1*output.shape[0])
        ).tolist()
        
        output_dict = {
           stop_id:sequence.index(i) for i, stop_id in enumerate(stop_ids)
        }
        
        # invalid += 1
    else: 
        output_dict = {
           stop_id:sequence[i] for i, stop_id in enumerate(stop_ids)
        }

    # write to file
    async with aiof.open(OUTPUT_PATH, mode='a') as file:
        data = json.load(file)
        data.update({
            route_id:{
                'proposed': output_dict
            }
        })
        file.seek(0)
        json.dump(data, file)

# Run tasks
loop = asyncio.get_event_loop()
loop.run_until_complete(
    asyncio.gather(
        *[solve(task) for task in tasks]
    )
)
loop.close()

# Finish Apply
print("Done!")
