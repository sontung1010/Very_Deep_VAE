import torch
import numpy as np
import os
import subprocess


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
