import torch
import json
import numpy as np
import os
import logging
import subprocess
import shutil



def check_nans(forward_result):
    distortion_nans = torch.isnan(forward_result['distortion']).sum()
    rate_nans = torch.isnan(forward_result['rate']).sum()
    return_dict = {
        'rate_nans': 0 if rate_nans == 0 else 1,
        'distortion_nans': 0 if distortion_nans == 0 else 1
    }
    return return_dict

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
    # use AdamW optimizer
    optimizer = torch.optim.AdamW(
        params=vae.parameters(),
        weight_decay=H.wd,
        lr=H.lr,
        betas=(H.adam_beta1, H.adam_beta2)
    )
    # use schedular with linear warmup
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_function(H.warmup_iters)
    )
    # this is for restarting training
    if H.restore_optimizer_path:
        optimizer_state = torch.load(H.restore_optimizer_path, map_location='cpu')
        optimizer.load_state_dict(optimizer_state)
        cur_eval_loss, iterate, starting_epoch = float('inf'), 0, 0
    else:
        cur_eval_loss, iterate, starting_epoch = float('inf'), 0, 0

    logger.info(f"Starting at epoch {starting_epoch}, iterate {iterate}, eval loss {cur_eval_loss}")
    return optimizer, scheduler, cur_eval_loss, iterate, starting_epoch

##====== change ======

def update_ema(vae, ema_vae, ema_rate):
    for p1, p2 in zip(vae.parameters(), ema_vae.parameters()):
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))


def saving_model(epoch, path, vae, ema_vae, optimizer, H):
    log_from = os.path.join(H.save_dir, 'log.jsonl')
    log_to = f'{os.path.dirname(path)}/{os.path.basename(path)}-log.jsonl'
    torch.save(vae.state_dict(), f'{path}-model'+str(epoch)+'.th')
    torch.save(ema_vae.state_dict(), f'{path}-model-ema'+str(epoch)+'.th')
    torch.save(optimizer.state_dict(), f'{path}-opt'+str(epoch)+'.th')
    shutil.copy(log_from, log_to)


def accumulate_stats(stats, frequency):
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
