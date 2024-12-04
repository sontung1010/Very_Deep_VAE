import torch
import numpy as np
import argparse
import os
import json
import subprocess
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import logger, maybe_download
from data import mkdir_p
from vae import VAE


def update_ema(vae, ema_vae, ema_rate):
    for p1, p2 in zip(vae.parameters(), ema_vae.parameters()):
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))


def save_model(path, vae, ema_vae, optimizer, H):
    torch.save(vae.state_dict(), f'{path}-model.th')
    torch.save(ema_vae.state_dict(), f'{path}-model-ema.th')
    torch.save(optimizer.state_dict(), f'{path}-opt.th')
    from_log = os.path.join(H.save_dir, 'log.jsonl')
    to_log = f'{os.path.dirname(path)}/{os.path.basename(path)}-log.jsonl'
    subprocess.check_output(['cp', from_log, to_log])


def accumulate_stats(stats, frequency):
    z = {}
    for k in stats[-1]:
        if k in ['distortion_nans', 'rate_nans', 'skipped_updates', 'gcskip']:
            z[k] = np.sum([a[k] for a in stats[-frequency:]])
        elif k == 'grad_norm':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            z[k] = np.max(finites) if len(finites) > 0 else 0.0
        elif k == 'elbo':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            z['elbo'] = np.mean(vals)
            z['elbo_filtered'] = np.mean(finites)
        elif k == 'iter_time':
            z[k] = np.mean([a[k] for a in stats[-frequency:]]) if len(stats) >= frequency else stats[-1][k]
        else:
            z[k] = np.mean([a[k] for a in stats[-frequency:]])
    return z


def linear_warmup(warmup_iters):
    def f(iteration):
        return 1.0 if iteration > warmup_iters else iteration / warmup_iters
    return f


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


def restore_log(path):
    loaded = [json.loads(l) for l in open(path)]
    try:
        cur_eval_loss = min([z['elbo'] for z in loaded if 'type' in z and z['type'] == 'eval_loss'])
    except ValueError:
        cur_eval_loss = float('inf')
    starting_epoch = max([z['epoch'] for z in loaded if 'type' in z and z['type'] == 'train_loss'])
    iterate = max([z['step'] for z in loaded if 'type' in z and z['type'] == 'train_loss'])
    return cur_eval_loss, iterate, starting_epoch


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


def load_opt(H, vae, logprint):
    optimizer = torch.optim.AdamW(vae.parameters(), weight_decay=H.wd, lr=H.lr, betas=(H.adam_beta1, H.adam_beta2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup(H.warmup_iters))

    if H.restore_optimizer_path:
        optimizer.load_state_dict(torch.load(H.restore_optimizer_path, map_location='cpu'))
    if H.restore_log_path:
        cur_eval_loss, iterate, starting_epoch = restore_log(H.restore_log_path)
    else:
        cur_eval_loss, iterate, starting_epoch = float('inf'), 0, 0
    logprint('starting at epoch', starting_epoch, 'iterate', iterate, 'eval loss', cur_eval_loss)
    return optimizer, scheduler, cur_eval_loss, iterate, starting_epoch