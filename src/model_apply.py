import logging
import os, json
import multiprocessing
import typing
import concurrent.futures

import torch
import numpy as np

import model.dataset as dataset
import model.search as search
import utils

# Set number of working cpu threads    
# https://jdhao.github.io/2020/07/06/pytorch_set_num_threads/
# https://github.com/pytorch/pytorch/issues/7087
torch.set_num_threads(multiprocessing.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())

# Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data/model_apply_outputs")
MODEL_PATH = os.path.join(BASE_DIR, 'data/model_build_outputs/best.script.pth')
OUTPUT_PATH = os.path.join(BASE_DIR, "data/model_apply_outputs/proposed_sequences.json")


def solve(task: search.Task):
    """Solve final sequence"""

    # Unpack arrays from task
    MAX_TIME = 24*3600*100
    route_id = task.route_id
    input = task.input
    output = task.output
    stop_ids = task.stop_ids
    station_id = task.station_id

    # Search problem initialization
    start_node = stop_ids.index(station_id)
    df_time = np.mean(input[:,:,-4].numpy(),axis=1) # TODO
    df_prob = (-output/10)**3*5 # top 1                TODO
    df_dist = (np.multiply(df_time,df_prob))         # TODO
    total_stops = len(df_dist)
    
    # Sequence
    sequence = search.or_search(df_dist, start_node, MAX_TIME, total_stops)

    if sequence is None:

        # Beam Search (greedy-decode)
        sequence = search.beam_search(
            start_node=start_node, weight_matrix=np.exp(output)*50, num_beam=1 
        ).tolist()
        
        output_dict = {
           stop_id:sequence.index(i) for i, stop_id in enumerate(stop_ids)
        }
        
    else: 
        output_dict = {
           stop_id:sequence[i] for i, stop_id in enumerate(stop_ids)
        }
    
    return route_id, output_dict


def main():
    """Model Apply"""

    # Load Model
    logging.info('Load Model...')

    model = torch.jit.load(MODEL_PATH)

    # Load Input Data
    logging.info('Reading Input Data...')

    datasets = dataset.get_dataset(["apply"], os.path.join(BASE_DIR, 'data'))
    apply_dataset = datasets["apply"]
    collate_fn = dataset.get_collate_fn(stage="apply", params=None)
    dataloader = dataset.DataLoader(apply_dataset, batch_size=1, collate_fn=collate_fn)

    logging.info("Apply dataset created.")

    # Model Apply Output
    tasks : typing.List[search.Task]= []

    # Loop through apply datasets
    logging.info("Runing model inference...")

    for batch in dataloader:

        # load batch data
        inputs = batch['inputs']              # (n, max_num_stops, max_num_stops, num_1d_features + num_2d_features)
        input_0ds = batch['input_0ds']        # (n, num_0d_features)
        masks = batch['masks']                # (n, max_num_stops, max_num_stops)
        route_id = batch["route_ids"][0]      # str
        station_id = batch['station_ids'][0]  # str
        stop_ids = batch['stop_ids'][0]       # str[]
        num_stops = batch['num_stops'][0]     # int
        
        # prediction
        output = model(inputs, input_0ds, masks)  # (num_stops, num_stops)

        # append to tasks list
        tasks.append(search.Task(
            route_id=route_id,
            stop_ids=stop_ids,
            station_id=station_id,
            num_stops=num_stops,
            input=inputs.squeeze(0).detach().numpy(),
            output=output.squeeze(0).detach().numpy()
        ))
    
    logging.info("Model inference completed.")

    # Run or-search
    logging.info("Running sequence search...")

    sequences = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for route_id, output_dict in executor.map(solve, tasks):
            sequences[route_id] = {'proposed': output_dict}
    
    logging.info("Sequence search completed.")

    # Save to json
    with open(OUTPUT_PATH, "w") as file:
        json.dump(sequences, file)
    
    logging.info(f"Sequence saved to file: {OUTPUT_PATH}")


if __name__ == "__main__":

    # Set the logger
    utils.set_logger(os.path.join(OUTPUT_DIR, 'apply.log'))

    # RUn model apply
    main()
    logging.info("Done!")
