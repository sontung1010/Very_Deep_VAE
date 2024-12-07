import torch
import numpy as np
import argparse
import os
from set_hyperparameter import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from data import mkdir_from_path
from vae import VAE


def setup_save_dirs(H):
    H.save_dir = os.path.join(H.save_dir, H.desc)
    mkdir_from_path(H.save_dir)
    H.logdir = os.path.join(H.save_dir, 'log')


def hyperparameter_setting(logger):
    ## recieved logger from the training file
    # =======================================================================
    hyperparameters = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    parse_args_and_update_hparams(hyperparameters, parser, s=None)
    setup_save_dirs(hyperparameters)
    for i, k in enumerate(sorted(hyperparameters)):
        try:
            logger.info("type:hparam, key= " + str(k) + " value= " + str(hyperparameters[k]))
        except:
            logger.info("minor errors on parameter loading")
    ## seed selection
    # =======================================================================
    np.random.seed(hyperparameters.seed)
    torch.cuda.manual_seed(hyperparameters.seed)
    torch.manual_seed(hyperparameters.seed)
    logger.info('training model ' + str(hyperparameters.desc) + ' dataset: ' + str(hyperparameters.dataset))
    return hyperparameters


def restore_params(model, path, map_cpu=False):
    ## This is for re initialization of the training
    # =======================================================================
    state_dict = torch.load(path, map_location='cpu' if map_cpu else None)
    model.load_state_dict(state_dict)


def load_model_custom(parameter, logger):
    ## Initialization of the training
    # =======================================================================
    vae_model = VAE(parameter)
    if parameter.restore_path:
        logger.info(f'Restoring VAE, {parameter.restore_path}')
        restore_params(vae_model, parameter.restore_path, map_cpu=True)

    ## Initialization of the training (ema_vae)
    # =======================================================================
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
    logger.info("total_parameter num= "+str(total_params) + ", readable= " + str(f'{total_params:,}'))
    return vae_model, ema_vae_model
