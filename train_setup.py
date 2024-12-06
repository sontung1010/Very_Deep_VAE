import torch
import numpy as np
import argparse
import os
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import logger, maybe_download
from data import mkdir_p
from vae import VAE


def setup_save_dirs(H):
    H.save_dir = os.path.join(H.save_dir, H.desc)
    mkdir_p(H.save_dir)
    H.logdir = os.path.join(H.save_dir, 'log')


def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    parse_args_and_update_hparams(H, parser, s=s)
    setup_save_dirs(H)
    logprint = logger(H.logdir)
    for i, k in enumerate(sorted(H)):
        logprint(type='hparam', key=k, value=H[k])
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)
    logprint('training model', H.desc, 'on', H.dataset)
    return H, logprint


def restore_params(model, path, map_cpu=False):
    state_dict = torch.load(path, map_location='cpu' if map_cpu else None)
    model.load_state_dict(state_dict)





def load_vaes(H, logprint):
    vae = VAE(H)
    if H.restore_path:
        logprint(f'Restoring vae from {H.restore_path}')
        restore_params(vae, H.restore_path, map_cpu=True)

    ema_vae = VAE(H)
    if H.restore_ema_path:
        logprint(f'Restoring ema vae from {H.restore_ema_path}')
        restore_params(ema_vae, H.restore_ema_path, map_cpu=True)
    else:
        ema_vae.load_state_dict(vae.state_dict())
    ema_vae.requires_grad_(False)

    vae = vae.cuda()
    ema_vae = ema_vae.cuda()

    total_params = sum(np.prod(p.shape) for p in vae.parameters())
    logprint(total_params=total_params, readable=f'{total_params:,}')
    return vae, ema_vae
