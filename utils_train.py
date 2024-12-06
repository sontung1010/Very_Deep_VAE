import torch
import logging

def check_nans(forward_result):
    distortion_nans = torch.isnan(forward_result['distortion']).sum()
    rate_nans = torch.isnan(forward_result['rate']).sum()
    return_dict = {
        'rate_nans': 0 if rate_nans == 0 else 1,
        'distortion_nans': 0 if distortion_nans == 0 else 1
    }
    return return_dict


def create_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger