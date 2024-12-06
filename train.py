import numpy as np
import imageio
import os
import time
import torch
from torch.utils.data import DataLoader
from data import set_up_data
from utils import get_cpu_stats_over_ranks
from train_helpers import set_up_hyperparams, load_vaes, load_opt, accumulate_stats, save_model, update_ema
from utils_train import check_nans, create_logger


# Reconstructing code

def eval_step(data_input, target, ema_vae):
    with torch.no_grad():
        stats = ema_vae.forward(data_input, target)
    stats = get_cpu_stats_over_ranks(stats)
    return stats


def get_sample_for_visualization(data, preprocess_fn, num, dataset):
    for x in DataLoader(data, batch_size=num):
        break
    orig_image = (x[0] * 255.0).to(torch.uint8).permute(0, 2, 3, 1) if dataset == 'ffhq_1024' else x[0]
    preprocessed = preprocess_fn(x)[0]
    return orig_image, preprocessed


def train_main(H, data_train, data_valid, preprocess_fn, vae, ema_vae, logprint):
    logger = create_logger("Training Logger")
    optimizer, scheduler, cur_eval_loss, iterate, starting_epoch = load_opt(H, vae, logprint)
    viz_batch_original, viz_batch_processed = get_sample_for_visualization(data_valid, preprocess_fn, H.num_images_visualize, H.dataset)
    early_evals = set([1] + [2 ** exp for exp in range(3, 14)])
    stats = []
    iters_since_starting = 0

    H.ema_rate = torch.as_tensor(H.ema_rate).cuda()

    for epoch in range(starting_epoch, H.num_epochs):
        print('\n\n')
        print("Starting epoch", str(epoch))
        traindata_loader = DataLoader(
            data_train,
            batch_size=H.n_batch,
            drop_last=True,
            pin_memory=True,
            shuffle=True
        )

        for x in traindata_loader:
            ## Starting single step
            # =======================================================================
            data_input, target = preprocess_fn(x)

            step_start = time.time()
            
            # initialize gradient for back prop
            vae.zero_grad()

            # Folowing output is from vae forward function
            # three information is included in this forward_result
            # {elbo, distortion, rate}
            forward_result = vae.forward(data_input, target)

            # using elbo loss fucntion
            forward_result['elbo'].backward()

            # limiting gradient value to H.grad_clip
            grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), H.grad_clip).item()

            # check for norm value in distortion and rate
            return_dict = check_nans(forward_result)
            forward_result.update(return_dict)
            

            skipped_updates = 1
            # only update if no rank has a nan and if the grad norm is below a specific threshold
            if forward_result['distortion_nans'] == 0 and forward_result['rate_nans'] == 0 and (H.skip_threshold == -1 or grad_norm < H.skip_threshold):
                optimizer.step()
                skipped_updates = 0
                update_ema(vae, ema_vae, H.ema_rate)

            step_end = time.time()
            step_duration = step_end - step_start
            forward_result.update(skipped_updates=skipped_updates, iter_time=step_duration, grad_norm=grad_norm)
            stats.append(forward_result)

            ## End the step
            # =======================================================================

            scheduler.step()
            if iterate % H.iters_per_print == 0 or iters_since_starting in early_evals:
                logger.info(model=H.desc, type='train_loss', lr=scheduler.get_last_lr()[0], epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))

            if iterate % H.iters_per_images == 0 or (iters_since_starting in early_evals and H.dataset != 'ffhq_1024') and H.rank == 0:
                write_images(H, ema_vae, viz_batch_original, viz_batch_processed, f'{H.save_dir}/samples-{iterate}.png', logprint)

            iterate += 1
            iters_since_starting += 1
            if iterate % H.iters_per_save == 0 and H.rank == 0:
                if np.isfinite(stats[-1]['elbo']):
                    logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))
                    fp = os.path.join(H.save_dir, 'latest')
                    logprint(f'Saving model@ {iterate} to {fp}')
                    save_model(fp, vae, ema_vae, optimizer, H)

            if iterate % H.iters_per_ckpt == 0 and H.rank == 0:
                save_model(os.path.join(H.save_dir, f'iter-{iterate}'), vae, ema_vae, optimizer, H)

        if epoch % H.epochs_per_eval == 0:
            valid_stats = evaluate(H, ema_vae, data_valid, preprocess_fn)
            logprint(model=H.desc, type='eval_loss', epoch=epoch, step=iterate, **valid_stats)


def evaluate(H, ema_vae, data_valid, preprocess_fn):
    stats_valid = []
    data_loader = DataLoader(data_valid, 
                             batch_size=H.n_batch, 
                             drop_last=True, 
                             pin_memory=True, 
                             shuffle=True) 

    for x in data_loader:
        data_input, target = preprocess_fn(x)
        stats_valid.append(eval_step(data_input, target, ema_vae))
    vals = [a['elbo'] for a in stats_valid]
    finites = np.array(vals)[np.isfinite(vals)]
    stats = dict(
        n_batches=len(vals),
        filtered_elbo=np.mean(finites),
        **{k: np.mean([a[k] for a in stats_valid]) for k in stats_valid[-1]}
    )

    return stats



def write_images(H, ema_vae, viz_batch_original, viz_batch_processed, fname, logprint):
    zs = [s['z'].cuda() for s in ema_vae.forward_get_latents(viz_batch_processed)]
    batches = [viz_batch_original.numpy()]
    mb = viz_batch_processed.shape[0]
    lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
    for i in lv_points:
        batches.append(ema_vae.forward_samples_set_latents(mb, zs[:i], t=0.1))
    for t in [1.0, 0.9, 0.8, 0.7][:H.num_temperatures_visualize]:
        batches.append(ema_vae.forward_uncond_samples(mb, t=t))
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *viz_batch_processed.shape[1:])).transpose([0, 2, 1, 3, 4]).reshape([n_rows * viz_batch_processed.shape[1], mb * viz_batch_processed.shape[2], 3])
    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)


def run_test_eval(H, ema_vae, data_test, preprocess_fn, logprint):
    print('evaluating')
    stats = evaluate(H, ema_vae, data_test, preprocess_fn)
    print('test results')
    for k in stats:
        print(k, stats[k])
    logprint(type='test_loss', **stats)


def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    vae, ema_vae = load_vaes(H, logprint)
    if H.test_eval:
        run_test_eval(H, ema_vae, data_valid_or_test, preprocess_fn, logprint)
    else:
        train_main(H, data_train, data_valid_or_test, preprocess_fn, vae, ema_vae, logprint)


if __name__ == "__main__":
    main()
