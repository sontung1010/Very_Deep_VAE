import torch
import numpy as np
import argparse
import os
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import maybe_download
from data import mkdir_p
from vae import VAE


def setup_save_dirs(H):
    H.save_dir = os.path.join(H.save_dir, H.desc)
    mkdir_p(H.save_dir)
    H.logdir = os.path.join(H.save_dir, 'log')


def hyperparameter_setting(logger):
    hyperparameters = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    parse_args_and_update_hparams(hyperparameters, parser, s=None)
    setup_save_dirs(hyperparameters)
    for i, k in enumerate(sorted(hyperparameters)):
        logger.info("type:hparam, key= " + str(k) + " value= " + str(hyperparameters[k]))
    np.random.seed(hyperparameters.seed)
    torch.cuda.manual_seed(hyperparameters.seed)
    torch.manual_seed(hyperparameters.seed)
    logger.info('training model ' + str(hyperparameters.desc) + ' on ' + str(hyperparameters.dataset))
    return hyperparameters


def restore_params(model, path, map_cpu=False):
    state_dict = torch.load(path, map_location='cpu' if map_cpu else None)
    model.load_state_dict(state_dict)


def load_model_custom(parameter, logger):
    vae_model = VAE(parameter)
    if parameter.restore_path:
        logger.info(f'Restoring VAE, {parameter.restore_path}')
        restore_params(vae_model, parameter.restore_path, map_cpu=True)

    ema_vae_model = VAE(parameter)
    if parameter.restore_ema_path:
        logger.info(f'Restoring ema VAE, {parameter.restore_ema_path}')
        restore_params(ema_vae_model, parameter.restore_ema_path, map_cpu=True)
    else:
        ema_vae_model.load_state_dict(vae_model.state_dict())
    ema_vae_model.requires_grad_(False)
    vae_model = vae_model.cuda()
    ema_vae_model = ema_vae_model.cuda()

    total_params = sum(np.prod(p.shape) for p in vae_model.parameters())
    logger.info("total_params= "+str(total_params) + ", readable= " + str(f'{total_params:,}'))
    return vae_model, ema_vae_model
