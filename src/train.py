import argparse
import logging
import multiprocessing
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
import model.net as net
import model.dataset as dataset
import model.loss as loss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=os.path.join(BASE_DIR, 'data'),
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default=os.path.join(BASE_DIR, 'experiments/base_model'),
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

def train_loop(model, optimizer, scheduler, criterion, metrics, params, model_dir, train_dataloader, writer, val_dataloader=None, restore_file=None, save_file=True):
    
    """Train the model.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) optional - a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        criterion: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional - name of file to restore from (without its extension .pth.tar)
        save_file: (bool) optional - boolean flag to save file
    """

    # Reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Logging
        logging.info(f"Epoch {epoch + 1}/{params.num_epochs}")

        # Train for one epoch (one full pass over the training set)
        train_metrics = train(model, optimizer, scheduler, criterion, train_dataloader, metrics, writer, params)

        # Evaluate for one epoch on validation set
        if val_dataloader is not None:
            # validation set metrics
            val_metrics = evaluate(model, criterion, val_dataloader, metrics, params)
            val_acc = val_metrics['accuracy']
            is_best = val_acc >= best_val_acc

            # write to tensorboard
            # issue with hparams https://github.com/pytorch/pytorch/issues/32651
            writer.add_scalar('Validation loss', val_metrics['loss'], epoch * len(train_dataloader))
            writer.add_scalar('Validation acc', val_metrics['accuracy'], epoch * len(train_dataloader))
            # writer.add_hparams(
            #     {'lr': params.learning_rate,"batch_size": params.batch_size},
            #     {'hparam/accuracy': val_metrics['accuracy'], 'hparam/loss': val_metrics['loss']}
            # )

        else:
            train_acc = train_metrics['accuracy']
            is_best = train_acc >= best_val_acc
        
        # Save latest val metrics in a json file in the model directory
        metrics_to_save = val_metrics if val_dataloader is not None else train_metrics
        metrics_type = "val" if val_dataloader is not None else "train"

        last_json_path = os.path.join(model_dir, f"metrics_{metrics_type}_last_weights.json")
        utils.save_dict_to_json(metrics_to_save, last_json_path)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc if val_dataloader is not None else train_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, f"metrics_{metrics_type}_best_weights.json")
            utils.save_dict_to_json(metrics_to_save, best_json_path)

        # Save weights
        if save_file:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict()},
                is_best=is_best,
                checkpoint=model_dir
            )


def train(model, optimizer, scheduler, criterion, dataloader, metrics, writer, params):
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
    epoch_counter = utils.Counter('epoch')
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader), unit="batch") as t:
        for i, batch in enumerate(dataloader):

            # Unpack batch, optionally get to cuda TODO
            inputs = batch['inputs'].to(params.device)       # (batch_m, max_num_stops, max_num_stops, num_1d_features + num_2d_features)
            input_0ds = batch['input_0ds'].to(params.device) # (batch_m, num_0d_features)
            targets = batch['targets'].to(params.device)     # (batch_m, max_num_stops)
            masks = batch['masks'].to(params.device)         # (batch_m, max_num_stops, max_num_stops)

            # Forward pass
            outputs = model(inputs, input_0ds, masks) # (batch_m, max_num_stops, max_num_stops)

            # Compute loss
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))

            # Backward pass
            optimizer.zero_grad() # clear previous grads
            loss.backward()

            # Gradient clipping - matain healthy grad 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.max_norm)

            # Update weights
            optimizer.step()

            # Scheduler step (step-wise scheduler, e.g., oneCycleLR)
            scheduler.step(loss.item())
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], (epoch_counter()-1) * len(dataloader) + i)

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # move to cpu
                outputs = outputs.data.cpu()
                targets = targets.data.cpu()

                # compute all metrics on this batch
                summary = {metric: metrics[metric](outputs, targets) for metric in metrics}
                summary['loss'] = loss.item()
                summ.append(summary)

                # write to tensorboard
                writer.add_scalar('Training loss', summary['loss'], (epoch_counter()-1) * len(dataloader) + i)
                writer.add_scalar('Training acc', summary['accuracy'], (epoch_counter()-1) * len(dataloader) + i)

            # Update the average loss
            loss_avg.update(loss.item())

            # Update tqdm
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # Compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    # Scheduler step (epoch-wise scheduler, e.g., reduceOnPlateou)
    # scheduler.step(loss_avg())
    # writer.add_scalar('Learning Rate', scheduler.get_lr()[0], (epoch_counter()-1) * len(dataloader) + i)

    return metrics_mean

def evaluate(model, criterion, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        criterion: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    Returns:
        metrics_mean: (dict) of float-castable validation-metric values (np.float, int, float, etc.)
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for batch in dataloader:

        # Unpack batch, optionally get to cuda TODO
        inputs = batch['inputs'].to(params.device)       # (batch_m, max_num_stops, max_num_stops, num_1d_features + num_2d_features)
        input_0ds = batch['input_0ds'].to(params.device) # (batch_m, num_0d_features)
        targets = batch['targets'].to(params.device)     # (batch_m, max_num_stops)
        masks = batch['masks'].to(params.device)         # (batch_m, max_num_stops, max_num_stops)

        # Compute model output
        outputs = model(inputs, input_0ds, masks) # (batch_m, max_num_stops, max_num_stops)
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
            
        # Move to cpu
        outputs = outputs.data.cpu()
        targets = targets.data.cpu()

        # compute all metrics on this batch
        summary = {metric: metrics[metric](outputs, targets) for metric in metrics}
        summary['loss'] = loss.item()
        summ.append(summary)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPU if available
    params.cuda = torch.cuda.is_available()
    params.device = torch.device("cuda" if params.cuda else "cpu")

    # Set default ignore_index
    if "ignore_index" not in params.dict:
        params.ignore_index = -100

    # Set number of working threads
    # https://jdhao.github.io/2020/07/06/pytorch_set_num_threads/
    # https://github.com/pytorch/pytorch/issues/7087
    torch.set_num_threads(multiprocessing.cpu_count())
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())

    # Seed everything
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Set tensorboard writer
    writer = SummaryWriter(os.path.join(args.model_dir, 'tb'))

    # Create the input data pipeline
    logging.info(f"Loading the datasets from {args.data_dir}...")

    # Creating dataloaders
    datasets = dataset.get_dataset(["build"], args.data_dir)
    build_dataset = datasets["build"]

    # train_set, val_set = torch.utils.data.random_split(build_dataset, [len(build_dataset)-200, 200])
    # train_set, val_set, others = torch.utils.data.random_split(build_dataset, [2000, 100, len(build_dataset)-2000-100])
    train_set, val_set, others = torch.utils.data.random_split(build_dataset, [params.train, params.val, len(build_dataset)-params.train-params.val])
    train_sampler = dataset.BucketSampler([route.num_stops for route in train_set], batch_size=params.batch_size, shuffle=True)
    val_sampler = dataset.BucketSampler([route.num_stops for route in val_set], batch_size=params.batch_size, shuffle=True)
    collate_fn = dataset.get_collate_fn(stage="build", params=params)

    train_loader = dataset.DataLoader(train_set, batch_sampler=train_sampler, collate_fn=collate_fn)
    val_loader = dataset.DataLoader(val_set, batch_sampler=val_sampler, collate_fn=collate_fn)
    logging.info("- done.")

    # Define the model, optimizer, and scheduler
    logging.info(f"Building model using params from {args.model_dir}")
    # model = net.RouteNet(router_embbed_dim=params.router_embbed_dim, num_routers=params.num_routers, dropout=params.dropout_rate).to(params.device)
    # model = net.RouteNetV2(router_embbed_dim=params.router_embbed_dim, num_routers=params.num_routers, dropout=params.dropout_rate).to(params.device)
    # model = net.RouteNetV3(router_embbed_dim=params.router_embbed_dim, num_routers=params.num_routers, dropout=params.dropout_rate).to(params.device)
    model = net.RouteNetV4(router_embbed_dim=params.router_embbed_dim, num_routers=params.num_routers, dropout=params.dropout_rate).to(params.device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=params.factor, patience=params.patience)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params.max_learning_rate, steps_per_epoch=len(train_loader), epochs=params.num_epochs, pct_start=params.anneal_start, div_factor=params.init_div_factor, final_div_factor=params.final_div_factor, anneal_strategy=params.strategy)
    
    # Define loss function and metrics
    # criterion =  torch.nn.NLLLoss(ignore_index=params.ignore_index)
    criterion =  loss.LabelSmoothingNLLLoss(ignore_index=params.ignore_index)
    metrics = net.metrics

    # Train the model
    logging.info(f"Starting training for {params.num_epochs} epoch(s)")
    train_loop(
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        criterion = criterion,
        metrics = metrics,
        params = params,
        model_dir = args.model_dir,
        writer=writer,
        train_dataloader = train_loader,
        val_dataloader = val_loader,
    )

    # Close tensorboard writer
    writer.close()
