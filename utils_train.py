import torch
import logging
import json

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
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
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