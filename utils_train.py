import torch
import numpy as np
import os
import logging
import shutil
import pandas as pd


def check_nans(forward_result):
    distortion_nans = torch.isnan(forward_result['distortion']).sum()
    rate_nans = torch.isnan(forward_result['rate']).sum()
    return_dict = {
        'rate_nans': 0 if rate_nans == 0 else 1,
        'distortion_nans': 0 if distortion_nans == 0 else 1
    }
    return return_dict

# this is for logging
# In reproduction we used more familiar logger
def create_logger(logger_name, log_file = "log_files.txt"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    return logger

def warmup_function(warmup_iters):
    return lambda iteration: max(1.0, iteration / warmup_iters) if iteration <= warmup_iters else 1.0

def load_optimizer(H, vae, logger):
    ## This code is for saving the model after iteration
    # =======================================================================
    optimizer = torch.optim.AdamW(
        params=vae.parameters(),
        weight_decay=H.wd,
        lr=H.lr,
        betas=(H.adam_beta1, H.adam_beta2)
    )
    # use schedular with linear warmup
    # =======================================================================
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_function(H.warmup_iters)
    )
    # this is for restarting training
    # =======================================================================
    if H.restore_optimizer_path:
        optimizer_state = torch.load(H.restore_optimizer_path, map_location='cpu')
        optimizer.load_state_dict(optimizer_state)
        cur_eval_loss, iterate, starting_epoch = float('inf'), 0, 0
    else:
        cur_eval_loss, iterate, starting_epoch = float('inf'), 0, 0

    logger.info(f"Starting at epoch {starting_epoch}, iterate {iterate}, eval loss {cur_eval_loss}")
    return optimizer, scheduler, cur_eval_loss, iterate, starting_epoch


def ema_vae_update(vae, ema_vae, ema_rate):
    vae_params = list(vae.parameters())
    ema_vae_params = list(ema_vae.parameters())
    for i in range(len(vae_params)):
        ema_value = ema_vae_params[i].data * ema_rate + vae_params[i].data * (1 - ema_rate)
        ema_vae_params[i].data = ema_value


def saving_model(epoch, path, vae, ema_vae, optimizer, H):
    ## This code is for saving the model after iteration
    # =======================================================================
    torch.save(vae.state_dict(), f'{path}_model'+str(epoch)+'.th')
    torch.save(ema_vae.state_dict(), f'{path}_model_ema'+str(epoch)+'.th')
    torch.save(optimizer.state_dict(), f'{path}_optimizer'+str(epoch)+'.th')


def add_row_train(df, epoch, setp, iteration_time, elbo, elbo_filtered, skipped_updates):
    # Create a new row as a dictionary
    new_row = {
        'epoch':epoch,
        'step': setp,
        'iteration_time': iteration_time,
        'elbo': elbo,
        'elbo_filtered': elbo_filtered,
        'skipped_updates': skipped_updates
    }
    # Append the new row to the DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df

def add_row_val(df, epoch, setp, elbo, elbo_filtered):
    # Create a new row as a dictionary
    new_row = {
        'epoch':epoch,
        'step': setp,
        'elbo': elbo,
        'elbo_filtered': elbo_filtered
    }
    # Append the new row to the DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def stats_batch_processing(stats, frequency):
    ## This code is for processign accumulated stats
    # Calculate the average value of the stats obtained in each iteration over a specific portion.
    # =======================================================================
    def safe_mean(values):
        return np.mean(values) if len(values) > 0 else 0.0
    def safe_max(values):
        return np.max(values) if len(values) > 0 else 0.0

    z = {}
    recent_stats = stats[-frequency:]
    for k in stats[-1]:
        values = [a[k] for a in recent_stats]
        if k in ['distortion_nans', 'rate_nans', 'skipped_updates', 'gcskip']:
            z[k] = np.sum(values) 
        elif k == 'grad_norm':
            finites = np.array(values)[np.isfinite(values)]
            z[k] = safe_max(finites) 
        elif k == 'elbo':
            finites = np.array(values)[np.isfinite(values)]
            z['elbo'] = safe_mean(values) 
            z['elbo_filtered'] = safe_mean(finites)
        elif k == 'iter_time' : z[k] = safe_mean(values) if len(stats) >= frequency else stats[-1][k]
        else : z[k] = safe_mean(values)  
    return z
