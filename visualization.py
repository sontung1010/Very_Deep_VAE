import numpy as np
import cv2
from torch.utils.data import DataLoader


def get_displaying_data(data, preprocess_fn, num, logger):
    dataloader = DataLoader(data, batch_size=num)
    x = next(iter(dataloader))
    original_image = x[0]
    logger.debug(f"Original image shape: {original_image.shape}")
    preprocessed_image = preprocess_fn(x)[0]
    logger.debug(f"Preprocessed image shape: {preprocessed_image.shape}")
    return original_image, preprocessed_image

def create_images(H, ema_vae, viz_batch_original, viz_batch_processed, fname, logger):
    zs = [s['z'].cuda() for s in ema_vae.forward_get_latents(viz_batch_processed)]
    batches = [viz_batch_original.numpy()]
    mb = viz_batch_processed.shape[0]
    
    lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
    for i in lv_points:
        batches.append(ema_vae.forward_samples_set_latents(mb, zs[:i], t=0.1))
    for t in [1.0, 0.9, 0.8, 0.7][:H.num_temperatures_visualize]:
        batches.append(ema_vae.forward_uncond_samples(mb, t=t))
    
    # Combine all batches into a single image
    n_rows = len(batches)
    im = (
        np.concatenate(batches, axis=0)
        .reshape((n_rows, mb, *viz_batch_processed.shape[1:]))
        .transpose([0, 2, 1, 3, 4])
        .reshape([n_rows * viz_batch_processed.shape[1], mb * viz_batch_processed.shape[2], 3])
    )
    
    im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    logger.info('Printing samples to ' + str(fname))
    cv2.imwrite(fname, im_bgr)

def formatting_text(stats):
    try:
        iteration_time = ' iteration_time: ' + str(stats['iter_time']) + ' '
        elbo_str = 'elbo: ' + str(stats['elbo'].item()) + ' '
        elbo_filtered_str = 'elbo_filtered: ' + str(stats['elbo_filtered'].item()) + ' '
        rate_nans = 'rate_nans: ' + str(stats['rate_nans'].item()) + ' '
        skipped_updates = 'skipped_updates: ' + str(stats['skipped_updates']) + ' '
    except : return "Failed logging."
    return iteration_time + elbo_str + elbo_filtered_str + rate_nans + skipped_updates

def formatting_text_validation(stats):
    try:
        n_batches_str = ' n_batches: ' + str(stats['n_batches']) + ' '
        elbo_str = 'elbo: ' + str(stats['elbo']) + ' '
        elbo_filtered_str = 'elbo_filtered: ' + str(stats['filtered_elbo']) + ' '
        rate_str = 'rate: ' + str(stats['rate']) + ' '
        distortion_str = 'distortion: ' + str(stats['distortion']) + ' '
    except : return "Failed logging."
    return n_batches_str + elbo_str + elbo_filtered_str + rate_str + distortion_str