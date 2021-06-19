import logging
import multiprocessing
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import utils
import model.net as net
import model.dataset as dataset
import model.loss as loss

# Get Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, "data/model_build_outputs")
PARAMS_PATH = os.path.join(BASE_DIR, 'src/config/params.json')


def train_loop(model, optimizer, scheduler, criterion, metrics, params, model_dir, train_dataloader, stats:dict):
    
    """Train the model.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        optimizer: (torch.optim) optimizer for parameters of model
        criterion: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
    """

    for epoch in range(params.num_epochs):
        # Logging
        logging.info(f"Epoch {epoch + 1}/{params.num_epochs}")

        # Train for one epoch (one full pass over the training set)
        train_metrics = train(model, optimizer, scheduler, criterion, train_dataloader, metrics, params, stats)

        # Evaluate for  one epoch on exponential moving average
        accuracy = stats['acc_moving_avg']()
        is_best = accuracy >= stats['best_accuracy']

        metrics_to_save = train_metrics
        metrics_type = "train"

        # Save most recent metrics
        last_json_path = os.path.join(model_dir, f"metrics_{metrics_type}_last_weights.json")
        utils.save_dict_to_json(metrics_to_save, last_json_path)

        # Update best metrics
        if is_best:
            logging.info("- Found new best accuracy")
            stats['best_accuracy'] = accuracy

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, f"metrics_{metrics_type}_best_weights.json")
            utils.save_dict_to_json(metrics_to_save, best_json_path)

        # Save model
        utils.save_torchscript(
            model=model,
            is_best=is_best,
            model_dir=OUTPUT_DIR
        )


def train(model, optimizer, scheduler, criterion, dataloader, metrics, params, stats:dict):
    """Train the model for one epoch (num_steps in batch model on `num_steps` batches)
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        scheduler: (torch.optim) scheduler to dynamically update learning rate
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    Returns:
        metrics_mean: (dict) of float-castable training-metric values (np.float, int, float, etc.)
    """

    # Set model to train mode
    model.train()

    # Summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader), unit="batch") as t:
        for i, batch in enumerate(dataloader):

            # Unpack batch, optionally get to cuda TODO
            inputs = batch['inputs']        # (batch_m, max_num_stops, max_num_stops, num_1d_features + num_2d_features)
            input_0ds = batch['input_0ds']  # (batch_m, num_0d_features)
            targets = batch['targets']      # (batch_m, max_num_stops)
            masks = batch['masks']          # (batch_m, max_num_stops, max_num_stops)

            # Forward pass
            outputs = model(inputs, input_0ds, masks) # (batch_m, max_num_stops, max_num_stops)

            # Compute loss
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), 
                targets.reshape(-1), 
            )

            # Backward pass
            optimizer.zero_grad() # clear previous grads
            loss.backward()

            # Gradient clipping - matain healthy grad 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.max_norm)

            # Update weights
            optimizer.step()

            # Scheduler step (step-wise scheduler, e.g., oneCycleLR)
            scheduler.step()

            # Evaluate summaries only once in a while
            summary = {metric: metrics[metric](outputs, targets) for metric in metrics}
            summary['loss'] = loss.item()
            summ.append(summary)

            # Update the average loss
            loss_avg.update(summary['loss'])
            stats['loss_moving_avg'].update(summary['loss'])
            stats['acc_moving_avg'].update(summary['accuracy'])

            # Save model
            if i % (len(dataloader)//6) == 0:
                accuracy = stats['acc_moving_avg']()
                is_best = accuracy >= stats['best_accuracy']

                metrics_to_save = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
                metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_to_save.items())
                metrics_type = "train"

                logging.info("- Train metrics: " + metrics_string)

                # Save most recent metrics
                last_json_path = os.path.join(OUTPUT_DIR, f"metrics_{metrics_type}_last_weights.json")
                utils.save_dict_to_json(metrics_to_save, last_json_path)

                # Update best metrics
                if is_best:
                    logging.info("- Found new best accuracy")
                    stats['best_accuracy'] = accuracy

                    # Save best val metrics in a json file in the model directory
                    best_json_path = os.path.join(OUTPUT_DIR, f"metrics_{metrics_type}_best_weights.json")
                    utils.save_dict_to_json(metrics_to_save, best_json_path)

                # Save model
                utils.save_torchscript(
                    model=model,
                    is_best=is_best,
                    model_dir=OUTPUT_DIR
                )
                

            # Update tqdm
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # Compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    return metrics_mean


if __name__ == '__main__':

    # Load the parameters from configuration json file
    params = utils.Params(PARAMS_PATH)

    # Set the logger
    utils.set_logger(os.path.join(OUTPUT_DIR, 'train.log'))

    # Set default ignore_index
    if "ignore_index" not in params.dict:
        params.ignore_index = -100


    # Set number of working threads
    # https://jdhao.github.io/2020/07/06/pytorch_set_num_threads/
    # https://github.com/pytorch/pytorch/issues/7087
    torch.set_num_threads(multiprocessing.cpu_count())
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())

    # Creating datasets, sampler, dataloaders
    logging.info(f"Loading the datasets from {DATA_DIR}...")

    datasets = dataset.get_dataset(["build"], DATA_DIR)
    build_dataset = datasets["build"]
    collate_fn = dataset.get_collate_fn(stage="build", params=params)
    train_sampler = dataset.BucketSampler([route.num_stops for route in build_dataset], batch_size=params.batch_size, shuffle=True)
    train_loader = dataset.DataLoader(build_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
   
    logging.info("- done.")

    # Define the model, optimizer, and scheduler
    logging.info(f"Building model using params from {PARAMS_PATH}")
    
    # (1) model
    model = net.RouteNetV5(
        router_embbed_dim=params.router_embbed_dim, 
        num_routers=params.num_routers, 
        num_heads=params.num_heads, 
        num_groups=params.num_groups, 
        contraction_factor=params.contraction_factor, 
        dropout=params.dropout_rate
    )
    
    # (2) optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=5e-5, 
        max_lr=1e-3, 
        step_size_up=10, 
        step_size_down=None, 
        mode='exp_range', 
        gamma=0.999, 
        scale_fn=None, 
        scale_mode='cycle', 
        cycle_momentum=False,
        base_momentum=0.8, 
        max_momentum=0.9, 
        last_epoch=-1, 
        verbose=False
    )
    
    # Define loss function and metrics
    criterion =  loss.LabelSmoothingNLLLoss(ignore_index=params.ignore_index)
    metrics = net.metrics

    # Train the model
    logging.info(f"Starting training for {params.num_epochs} epoch(s)")

    stats = {
        'best_accuracy' : 0.0,
        'loss_moving_avg' : utils.MovingAverage(),
        'acc_moving_avg' : utils.MovingAverage(),
    }
    
    train_loop(
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        criterion = criterion,
        metrics = metrics,
        params = params,
        model_dir = OUTPUT_DIR,
        train_dataloader = train_loader,
        stats=stats
    )

    logging.info(f"Done")
